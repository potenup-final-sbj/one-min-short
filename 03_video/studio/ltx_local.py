from __future__ import annotations

import gc
import hashlib
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Callable


logger = logging.getLogger("uvicorn.error").getChild(__name__)

LTX_VIDEO_MODEL = os.getenv("LTX_VIDEO_MODEL", "Lightricks/LTX-Video-0.9.5")
LTX_WIDTH = 288
LTX_HEIGHT = 512
LTX_NUM_FRAMES = 49
LTX_FPS = 24
LTX_NUM_INFERENCE_STEPS = 25
LTX_GUIDANCE_SCALE = 5.0
LTX_DECODE_TIMESTEP = 0.05
LTX_DECODE_NOISE_SCALE = 0.025
LTX_NEGATIVE_PROMPT = (
    "worst quality, low quality, blurry, jittery, distorted, inconsistent motion, "
    "distorted face, deformed hands, duplicate person, text, logo, watermark"
)


class LtxConfigurationError(RuntimeError):
    pass


class LtxLocalClient:
    """Local LTX image-to-video client with one pipeline per render request."""

    provider_name = "Local LTX Image-to-Video"
    artifact_name = "ltx"

    def __init__(self, base_dir: Path, log: Callable[[str], None] | None = None):
        self.base_dir = base_dir
        self.cache_dir = base_dir / "outputs" / "ltx_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log = log or (lambda _message: None)
        self._pipeline = None

    def _log_info(self, message: str) -> None:
        logger.info(message)
        self.log(message)

    @staticmethod
    def is_available() -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            logger.warning("LTX CUDA 상태를 확인할 수 없습니다: torch가 설치되지 않았습니다.")
            return False

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            import torch
            from diffusers import LTXImageToVideoPipeline
        except ImportError as exc:
            logger.exception("LTX 파이프라인 의존성을 불러오지 못했습니다.")
            raise LtxConfigurationError(
                "로컬 LTX 의존성이 없습니다. torch, diffusers, transformers, "
                "accelerate, imageio, sentencepiece를 설치하세요."
            ) from exc

        if not torch.cuda.is_available():
            logger.error("LTX 모델을 로드할 수 없습니다: CUDA GPU를 사용할 수 없습니다.")
            raise LtxConfigurationError(
                "로컬 LTX 영상 생성에는 CUDA를 지원하는 NVIDIA GPU가 필요합니다."
            )

        started_at = time.perf_counter()
        self._log_info(f"로컬 LTX 모델 로드 시작: {LTX_VIDEO_MODEL}")
        pipeline = LTXImageToVideoPipeline.from_pretrained(
            LTX_VIDEO_MODEL,
            torch_dtype=torch.bfloat16,
        )
        pipeline.enable_model_cpu_offload()
        pipeline.vae.enable_tiling()
        self._pipeline = pipeline
        self._log_info(
            f"로컬 LTX 모델 로드 완료: model={LTX_VIDEO_MODEL}, "
            f"elapsed={time.perf_counter() - started_at:.2f}s"
        )
        return pipeline

    @staticmethod
    def _cache_key(image_path: Path, prompt: str, seed: int) -> str:
        digest = hashlib.sha256()
        digest.update(image_path.read_bytes())
        digest.update(prompt.encode("utf-8"))
        digest.update(
            (
                f"{LTX_VIDEO_MODEL};{seed};{LTX_WIDTH}x{LTX_HEIGHT};"
                f"frames={LTX_NUM_FRAMES};fps={LTX_FPS};"
                f"steps={LTX_NUM_INFERENCE_STEPS};guidance={LTX_GUIDANCE_SCALE};"
                f"decode={LTX_DECODE_TIMESTEP},{LTX_DECODE_NOISE_SCALE}"
            ).encode("utf-8")
        )
        return digest.hexdigest()[:24]

    def generate(
        self,
        image_path: Path,
        motion_prompt: str,
        output_path: Path,
        *,
        seed: int,
    ) -> dict:
        if not image_path.is_file():
            logger.error("LTX 입력 이미지를 찾을 수 없습니다: %s", image_path)
            raise FileNotFoundError(f"LTX 입력 이미지를 찾을 수 없습니다: {image_path}")

        cache_path = self.cache_dir / f"{self._cache_key(image_path, motion_prompt, seed)}.mp4"
        if cache_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, output_path)
            self._log_info(
                f"LTX 캐시 사용: cache={cache_path.name}, output={output_path.name}, seed={seed}"
            )
            return {
                "provider": self.provider_name,
                "model": LTX_VIDEO_MODEL,
                "seed": seed,
                "prompt": motion_prompt,
                "cached": True,
                "quota_fallback": False,
            }

        try:
            import torch
            from diffusers.utils import export_to_video, load_image

            started_at = time.perf_counter()
            pipeline = self._load_pipeline()
            generator = torch.Generator(device="cuda").manual_seed(seed)
            self._log_info(
                f"로컬 LTX 영상 생성 시작: input={image_path.name}, "
                f"output={output_path.name}, seed={seed}"
            )
            frames = pipeline(
                image=load_image(str(image_path)),
                prompt=motion_prompt,
                negative_prompt=LTX_NEGATIVE_PROMPT,
                width=LTX_WIDTH,
                height=LTX_HEIGHT,
                num_frames=LTX_NUM_FRAMES,
                frame_rate=LTX_FPS,
                num_inference_steps=LTX_NUM_INFERENCE_STEPS,
                guidance_scale=LTX_GUIDANCE_SCALE,
                decode_timestep=LTX_DECODE_TIMESTEP,
                decode_noise_scale=LTX_DECODE_NOISE_SCALE,
                generator=generator,
            ).frames[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            export_to_video(frames, str(output_path), fps=LTX_FPS)
        except LtxConfigurationError:
            raise
        except Exception as exc:
            logger.exception(
                "로컬 LTX 영상 생성 실패: input=%s, output=%s, seed=%s",
                image_path.name,
                output_path.name,
                seed,
            )
            raise RuntimeError(f"로컬 LTX 영상 생성에 실패했습니다: {exc}") from exc

        if not output_path.is_file():
            logger.error("LTX 출력 MP4가 생성되지 않았습니다: %s", output_path)
            raise RuntimeError(f"로컬 LTX가 유효한 MP4를 생성하지 않았습니다: {output_path}")
        shutil.copy2(output_path, cache_path)
        self._log_info(
            f"로컬 LTX 영상 생성 완료: output={output_path.name}, "
            f"size={output_path.stat().st_size}, seed={seed}, "
            f"elapsed={time.perf_counter() - started_at:.2f}s"
        )
        return {
            "provider": self.provider_name,
            "model": LTX_VIDEO_MODEL,
            "seed": seed,
            "prompt": motion_prompt,
            "cached": False,
            "quota_fallback": False,
        }

    def close(self) -> None:
        had_pipeline = self._pipeline is not None
        if self._pipeline is not None:
            del self._pipeline
            self._pipeline = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            logger.warning("LTX 메모리 정리 중 torch를 불러오지 못했습니다.")
        if had_pipeline:
            self._log_info("로컬 LTX 파이프라인 메모리 정리 완료")
