from __future__ import annotations

import gc
import os
from pathlib import Path


IMAGE_MODEL = os.getenv("SDXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
LIGHTNING_MODEL = os.getenv("SDXL_LIGHTNING_MODEL", "ByteDance/SDXL-Lightning")
LIGHTNING_CHECKPOINT = "sdxl_lightning_4step_unet.safetensors"


class SceneImageGenerationError(RuntimeError):
    pass


class SceneImageGenerator:
    """Generate one fresh story image with local SDXL-Lightning."""

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir

    @staticmethod
    def _prompt(story: dict) -> str:
        return "\n".join(
            (
                "Vertical 9:16 cinematic live-action Korean short drama still.",
                f"The user's exact story premise is: {story['logline']}",
                f"Hero composition: {story['hero_visual_prompt']}",
                f"Character and setting continuity: {story['visual_bible']}",
                f"Genre: {story['genre']}. Mood: {story['mood']}.",
                (
                    "Depict every essential subject from the premise literally. Photorealistic "
                    "people and environment, natural anatomy, expressive faces, cinematic "
                    "lighting, no text, no subtitles, no logo, no watermark."
                ),
            )
        )

    @staticmethod
    def _load_pipeline():
        try:
            import torch
            from diffusers import EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
            from huggingface_hub import hf_hub_download
            from safetensors.torch import load_file
        except ImportError as exc:
            raise SceneImageGenerationError(
                "로컬 이미지 의존성이 없습니다. torch, diffusers, transformers, "
                "accelerate, safetensors를 설치하세요."
            ) from exc

        if not torch.cuda.is_available():
            raise SceneImageGenerationError(
                "SDXL-Lightning 로컬 생성에는 CUDA를 지원하는 NVIDIA GPU가 필요합니다."
            )

        unet = UNet2DConditionModel.from_config(IMAGE_MODEL, subfolder="unet")
        checkpoint = hf_hub_download(LIGHTNING_MODEL, LIGHTNING_CHECKPOINT)
        unet.load_state_dict(load_file(checkpoint, device="cpu"))
        unet.to(dtype=torch.float16)
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            IMAGE_MODEL,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        pipeline.scheduler = EulerDiscreteScheduler.from_config(
            pipeline.scheduler.config, timestep_spacing="trailing"
        )
        pipeline.enable_attention_slicing()
        pipeline.enable_model_cpu_offload()
        return pipeline

    def generate(self, story: dict, output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "story_01.png"
        pipeline = None
        try:
            pipeline = self._load_pipeline()
            result = pipeline(
                self._prompt(story),
                num_inference_steps=4,
                guidance_scale=0,
                width=512,
                height=896,
                generator=None,
            )
            result.images[0].convert("RGB").save(output_path, format="PNG")
        except SceneImageGenerationError:
            raise
        except Exception as exc:
            raise SceneImageGenerationError(
                f"로컬 SDXL-Lightning 이미지 생성에 실패했습니다: {exc}"
            ) from exc
        finally:
            if pipeline is not None:
                del pipeline
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        return [output_path]
