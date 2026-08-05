from __future__ import annotations

from pathlib import Path

from huggingface_hub import InferenceClient, get_token


IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
class SceneImageGenerationError(RuntimeError):
    pass


class SceneImageGenerator:
    """Generate one fresh prompt-grounded still for the whole story."""

    def __init__(self, base_dir: Path):
        token = get_token()
        if not token:
            raise SceneImageGenerationError(
                "Hugging Face 로그인이 필요합니다. `hf auth login`을 먼저 실행하세요."
            )
        self.client = InferenceClient(provider="auto", api_key=token, timeout=180)

    @staticmethod
    def _prompt(story: dict, scene: dict, shot_number: int) -> str:
        premise = str(story["logline"]).strip()
        prompt_parts = [
                "Vertical 9:16 cinematic live-action Korean short drama still.",
                f"The user's exact story premise is: {premise}",
                (
                    "Depict that premise and this scene literally. The location must "
                    "match the described story world; do not substitute a generic setting."
                ),
                f"Character and setting continuity: {story['visual_bible']}",
                f"Genre: {story['genre']}. Mood: {story['mood']}.",
                f"Shot {shot_number}: {scene['visual_prompt']}",
                (
                    "Photorealistic people and environment, coherent production design, "
                    "natural anatomy, expressive faces, cinematic lighting, shallow depth "
                    "of field, no text, no subtitles, no logo, no watermark."
                ),
        ]
        return "\n".join(prompt_parts)

    @staticmethod
    def _negative_prompt() -> str:
        return "text, subtitles, logo, watermark, malformed anatomy, generic stock photo"

    def generate(self, story: dict, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = story["common_scenes"][0]
        prompt = self._prompt(story, scene, 1)
        seed = 3101
        output_path = output_dir / "story_01.png"

        try:
            image = self.client.text_to_image(
                prompt,
                negative_prompt=self._negative_prompt(),
                model=IMAGE_MODEL,
                width=768,
                height=1344,
                num_inference_steps=4,
                guidance_scale=3.5,
                seed=seed,
            )
        except Exception as exc:
            raise SceneImageGenerationError(
                "프롬프트 이미지 생성에 실패했습니다. Hugging Face "
                f"Inference Providers 권한·무료 크레딧을 확인하세요: {exc}"
            ) from exc

        image.convert("RGB").save(output_path, format="PNG")
        return [output_path]
