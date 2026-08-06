from __future__ import annotations

import hashlib
import json
import os
import gc
from pathlib import Path

STORY_MODEL = os.getenv("HF_STORY_MODEL", "Qwen/Qwen3-4B")
STORY_MAX_NEW_TOKENS = int(os.getenv("STORY_MAX_NEW_TOKENS", "1500"))
STORY_PROMPT_VERSION = "local-hf-qwen-nf4-thirty-second-v1"
ACCENTS = {
    "로맨스": "#ff4d8d",
    "스릴러": "#8b5cf6",
    "코미디": "#ffb703",
    "판타지": "#22d3ee",
}


class StoryGenerationError(RuntimeError):
    pass


SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "location": {"type": "string"},
        "speaker": {"type": "string"},
        "dialogue": {"type": "string"},
        "visual_prompt": {"type": "string"},
        "motion_prompt": {"type": "string"},
    },
    "required": [
        "title", "location", "speaker", "dialogue",
        "visual_prompt", "motion_prompt",
    ],
    "additionalProperties": False,
}

STORY_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "logline": {"type": "string"},
        "visual_bible": {"type": "string"},
        "hero_visual_prompt": {"type": "string"},
        "characters": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "voice": {"type": "string", "enum": ["female", "male"]},
                },
                "required": ["name", "role", "voice"],
                "additionalProperties": False,
            },
        },
        "common_scenes": {
            "type": "array", "minItems": 3, "maxItems": 3,
            "items": SCENE_SCHEMA,
        },
        "ending_a": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": SCENE_SCHEMA,
        },
        "ending_b": {
            "type": "array", "minItems": 1, "maxItems": 1,
            "items": SCENE_SCHEMA,
        },
    },
    "required": [
        "title", "logline", "visual_bible", "hero_visual_prompt", "characters",
        "common_scenes", "ending_a", "ending_b",
    ],
    "additionalProperties": False,
}


def _validate_scenes(story: dict, accent: str) -> dict:
    groups = (
        ("common_scenes", "common", 3, 6),
        ("ending_a", "ending_a", 1, 12),
        ("ending_b", "ending_b", 1, 12),
    )
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
    for field in ("visual_bible", "hero_visual_prompt"):
        if not isinstance(story.get(field), str) or not story[field].strip():
            raise StoryGenerationError(f"{field}가 비어 있습니다.")
    return story


def create_story(prompt: str, genre: str, mood: str) -> dict:
    cache_dir = Path(__file__).resolve().parents[1] / "outputs" / "story_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = hashlib.sha256(
        f"{STORY_PROMPT_VERSION}\n{STORY_MODEL}\n{prompt}\n{genre}\n{mood}".encode("utf-8")
    ).hexdigest()[:24]
    cache_path = cache_dir / f"{cache_key}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    instruction = f"""
Create a complete 30-second Korean vertical short-drama storyboard.

USER PREMISE: {prompt}
GENRE: {genre}
MOOD: {mood}

Rules:
- Write exactly 3 common scenes and exactly one scene for each A/B ending.
- Every character, location, event, and line must derive from USER PREMISE.
- Korean dialogue must fit naturally within each scene.
- visual_bible must describe recurring characters, wardrobe, era, and setting in English.
- hero_visual_prompt must be one detailed English image prompt representing the WHOLE premise.
- hero_visual_prompt must include every essential subject, animal, object, relationship, and location
  from USER PREMISE even if it appears later in the story.
- Each visual_prompt and motion_prompt must be in English.
- Do not introduce an office, CEO, former lover, or unrelated stock setting unless requested.
- Make ending A and ending B meaningfully different.
""".strip()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Korean screenwriter and storyboard director. "
                "Return only one valid JSON object with no markdown or commentary."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{instruction}\n\nThe JSON must match this schema exactly:\n"
                f"{json.dumps(STORY_SCHEMA, ensure_ascii=False)}"
            ),
        },
    ]
    model = None
    tokenizer = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        tokenizer = AutoTokenizer.from_pretrained(STORY_MODEL)
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            STORY_MODEL,
            quantization_config=quantization_config,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model.eval()
        rendered_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = tokenizer([rendered_prompt], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=STORY_MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                repetition_penalty=1.05,
            )
        output_ids = generated_ids[0][model_inputs.input_ids.shape[-1]:]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        json_start = content.find("{")
        json_end = content.rfind("}")
        if json_start < 0 or json_end < json_start:
            raise json.JSONDecodeError("JSON object not found", content, 0)
        story = json.loads(content[json_start : json_end + 1])
    except (ImportError, OSError, RuntimeError, TypeError, json.JSONDecodeError) as exc:
        raise StoryGenerationError(
            f"로컬 Hugging Face Qwen 대본 생성에 실패했습니다. "
            f"{STORY_MODEL} 다운로드와 GPU 메모리를 확인하세요: {exc}"
        ) from exc
    finally:
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    story["logline"] = prompt
    story["genre"] = genre
    story["mood"] = mood
    story["duration"] = {"common": 18, "ending": 12, "total": 30}
    story = _validate_scenes(story, ACCENTS.get(genre, "#ff4d8d"))
    cache_path.write_text(json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8")
    return story
