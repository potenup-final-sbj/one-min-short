from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from huggingface_hub import InferenceClient, get_token


IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
PROMPT_MODEL = "openai/gpt-oss-20b"


class SceneImageGenerationError(RuntimeError):
    pass


class SceneImageGenerator:
    """Generate and cache three prompt-grounded vertical drama stills."""

    def __init__(self, base_dir: Path):
        token = get_token()
        if not token:
            raise SceneImageGenerationError(
                "Hugging Face 로그인이 필요합니다. `hf auth login`을 먼저 실행하세요."
            )
        self.client = InferenceClient(provider="auto", api_key=token, timeout=180)
        self.cache_dir = base_dir / "outputs" / "image_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _translated_premise(self, story: dict) -> str:
        source = story["logline"].strip()
        cache_key = hashlib.sha256(
            f"{PROMPT_MODEL}\n{source}".encode("utf-8")
        ).hexdigest()[:24]
        cache_path = self.cache_dir / f"prompt_{cache_key}.txt"
        if cache_path.is_file():
            return cache_path.read_text(encoding="utf-8").strip()

        try:
            response = self.client.chat_completion(
                model=PROMPT_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Translate the user's Korean story premise into a literal, "
                            "visually specific English description for an image model. "
                            "Preserve the exact characters, setting, relationship, era, "
                            "and central event. Output only the English translation."
                        ),
                    },
                    {"role": "user", "content": source},
                ],
                max_tokens=600,
                temperature=0,
            )
            translated = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            raise SceneImageGenerationError(
                f"한국어 소재를 이미지 프롬프트로 변환하지 못했습니다: {exc}"
            ) from exc
        if not translated:
            raise SceneImageGenerationError("이미지 프롬프트 변환 결과가 비어 있습니다.")
        cache_path.write_text(translated, encoding="utf-8")
        return translated

    @staticmethod
    def _prompt(story: dict, translated_premise: str, shot_number: int) -> str:
        shot_directions = {
            1: (
                "Establish the characters and environment described by the premise. "
                "A visually clear opening moment that immediately communicates the setting."
            ),
            2: (
                "Show the central relationship and conflict from the premise developing. "
                "Medium cinematic shot, meaningful eye contact and dramatic tension."
            ),
            3: (
                "Show the emotional climax and a decisive confrontation implied by the premise. "
                "Intimate cinematic composition with strong but natural emotion."
            ),
        }
        return "\n".join(
            (
                "Vertical 9:16 cinematic live-action Korean short drama still.",
                f"Story premise: {translated_premise}",
                f"Genre: {story['genre']}. Mood: {story['mood']}.",
                f"Shot {shot_number}: {shot_directions[shot_number]}",
                (
                    "Photorealistic people and environment, coherent production design, "
                    "natural anatomy, expressive faces, cinematic lighting, shallow depth "
                    "of field, no text, no subtitles, no logo, no watermark."
                ),
            )
        )

    def generate(self, story: dict, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        translated_premise = self._translated_premise(story)

        for index in range(1, 4):
            prompt = self._prompt(story, translated_premise, index)
            cache_key = hashlib.sha256(
                f"{IMAGE_MODEL}\n{prompt}\nseed={3100 + index}".encode("utf-8")
            ).hexdigest()[:24]
            cache_path = self.cache_dir / f"{cache_key}.png"
            output_path = output_dir / f"story_{index:02}.png"

            if cache_path.is_file():
                shutil.copy2(cache_path, output_path)
                outputs.append(output_path)
                continue

            try:
                image = self.client.text_to_image(
                    prompt,
                    model=IMAGE_MODEL,
                    width=768,
                    height=1344,
                    num_inference_steps=4,
                    guidance_scale=3.5,
                    seed=3100 + index,
                )
            except Exception as exc:
                raise SceneImageGenerationError(
                    "프롬프트별 장면 이미지 생성에 실패했습니다. Hugging Face "
                    f"Inference Providers 권한·무료 크레딧을 확인하세요: {exc}"
                ) from exc

            image.convert("RGB").save(cache_path, format="PNG")
            shutil.copy2(cache_path, output_path)
            outputs.append(output_path)

        return outputs
