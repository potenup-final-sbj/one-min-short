from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from huggingface_hub import InferenceClient, get_token


STORY_MODEL = "openai/gpt-oss-20b"
STORY_PROMPT_VERSION = "thirty-second-v1"
ACCENTS = {
    "로맨스": "#ff4d8d",
    "스릴러": "#8b5cf6",
    "코미디": "#ffb703",
    "판타지": "#22d3ee",
}


class StoryGenerationError(RuntimeError):
    pass


def _extract_json(value: str) -> dict:
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StoryGenerationError(f"스토리 JSON을 해석하지 못했습니다: {exc}") from exc
    if not isinstance(result, dict):
        raise StoryGenerationError("스토리 생성 결과가 JSON 객체가 아닙니다.")
    return result


def _validate_scenes(story: dict, accent: str) -> dict:
    groups = (("common_scenes", "common", 3, 6), ("ending_a", "ending_a", 1, 12), ("ending_b", "ending_b", 1, 12))
    required = {
        "title", "location", "speaker", "dialogue", "visual_prompt", "motion_prompt"
    }
    for group_name, phase, expected_count, duration in groups:
        scenes = story.get(group_name)
        if not isinstance(scenes, list) or len(scenes) != expected_count:
            raise StoryGenerationError(
                f"{group_name} 장면 수가 {expected_count}개가 아닙니다."
            )
        for index, scene in enumerate(scenes, start=1):
            if not isinstance(scene, dict) or not required.issubset(scene):
                raise StoryGenerationError(f"{group_name} {index}번 장면 필드가 부족합니다.")
            scene["id"] = f"{phase}_{index:02}"
            scene["phase"] = phase
            scene["subtitle"] = str(scene["dialogue"])
            scene["duration"] = duration
            scene["accent"] = accent
    if not isinstance(story.get("characters"), list) or not story["characters"]:
        raise StoryGenerationError("등장인물 정보가 없습니다.")
    if not isinstance(story.get("visual_bible"), str) or not story["visual_bible"].strip():
        raise StoryGenerationError("캐릭터·배경 비주얼 바이블이 없습니다.")
    return story


def create_story(prompt: str, genre: str, mood: str) -> dict:
    token = get_token()
    if not token:
        raise StoryGenerationError(
            "Hugging Face 로그인이 필요합니다. `hf auth login`을 먼저 실행하세요."
        )

    cache_dir = Path(__file__).resolve().parents[1] / "outputs" / "story_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{STORY_PROMPT_VERSION}\n{STORY_MODEL}\n{prompt}\n{genre}\n{mood}".encode("utf-8")
    ).hexdigest()[:24]
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    schema = {
        "title": "Korean title, no more than 28 characters",
        "logline": "the exact user premise in Korean",
        "visual_bible": "English description of recurring characters, wardrobe, era, and setting for image consistency",
        "characters": [
            {"name": "Korean name", "role": "Korean role", "voice": "female or male"}
        ],
        "common_scenes": [
            {
                "title": "Korean scene title",
                "location": "Korean location",
                "speaker": "character name or 내레이션",
                "dialogue": "natural Korean narration or dialogue",
                "visual_prompt": "English cinematic still description grounded in the user premise",
                "motion_prompt": "English I2V character action and camera direction",
            }
        ],
        "ending_a": "same scene object format, exactly 1 scene",
        "ending_b": "same scene object format, exactly 1 scene",
    }
    instruction = f"""
Create a complete 30-second Korean vertical short-drama storyboard from the user's premise.

USER PREMISE: {prompt}
GENRE: {genre}
MOOD: {mood}

Return only one valid JSON object matching this shape:
{json.dumps(schema, ensure_ascii=False, indent=2)}

Rules:
- Write exactly 3 common_scenes scenes, exactly 1 ending_a scene, and exactly 1 ending_b scene.
- The three common scenes last 6 seconds each. Each alternative ending lasts 12 seconds.
- Every character, location, event, line, visual_prompt, and motion_prompt must derive from USER PREMISE.
- Do not introduce offices, company CEOs, former lovers, or an eight-year separation unless USER PREMISE explicitly asks for them.
- Keep the same protagonist appearance and wardrobe across all English visual prompts by following visual_bible.
- visual_prompt must describe the exact visible setting, characters, wardrobe, composition, lighting, and emotion.
- motion_prompt must describe actions visible from the still plus restrained camera movement suitable for image-to-video.
- Korean dialogue must be concise enough to speak within each scene.
- ending_a and ending_b must be meaningfully different choices.
- Do not include IDs, phase, duration, subtitle, or accent; the application adds them.
""".strip()

    client = InferenceClient(provider="auto", api_key=token, timeout=180)
    try:
        response = client.chat_completion(
            model=STORY_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Korean screenwriter and storyboard director. "
                        "Follow the requested JSON format exactly."
                    ),
                },
                {"role": "user", "content": instruction},
            ],
            max_tokens=5000,
            temperature=0.45,
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:
        raise StoryGenerationError(f"대본 생성 API 호출에 실패했습니다: {exc}") from exc
    if not content.strip():
        raise StoryGenerationError("대본 생성 결과가 비어 있습니다.")

    story = _extract_json(content)
    story["logline"] = prompt
    story["genre"] = genre
    story["mood"] = mood
    story["duration"] = {"common": 18, "ending": 12, "total": 30}
    story = _validate_scenes(story, ACCENTS.get(genre, "#ff4d8d"))
    cache_path.write_text(
        json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return story
