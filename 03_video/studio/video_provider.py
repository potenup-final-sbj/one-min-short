from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Protocol

from studio.ltx_local import LtxLocalClient
from studio.wan22_cloud import Wan22CloudClient


logger = logging.getLogger("uvicorn.error").getChild(__name__)

SUPPORTED_VIDEO_PROVIDERS = ("ltx", "wan22")


class VideoProviderConfigurationError(RuntimeError):
    pass


class VideoGenerator(Protocol):
    provider_name: str
    artifact_name: str

    def generate(
        self,
        image_path: Path,
        motion_prompt: str,
        output_path: Path,
        *,
        seed: int,
    ) -> dict: ...

    def close(self) -> None: ...


def validate_video_provider(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized not in SUPPORTED_VIDEO_PROVIDERS:
        supported = ", ".join(SUPPORTED_VIDEO_PROVIDERS)
        logger.error(
            "지원하지 않는 영상 공급자 설정: provider=%r, supported=%s",
            provider,
            supported,
        )
        raise VideoProviderConfigurationError(
            f"지원하지 않는 VIDEO_PROVIDER입니다: {provider!r}. 지원 값: {supported}"
        )
    return normalized


VIDEO_PROVIDER = validate_video_provider(os.getenv("VIDEO_PROVIDER", "ltx"))


def create_video_generator(base_dir: Path, provider: str = VIDEO_PROVIDER) -> VideoGenerator:
    selected = validate_video_provider(provider)
    logger.info("영상 공급자 선택: provider=%s", selected)
    if selected == "ltx":
        return LtxLocalClient(base_dir)
    return Wan22CloudClient()
