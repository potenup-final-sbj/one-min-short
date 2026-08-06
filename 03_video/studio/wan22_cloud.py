from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from gradio_client import Client, handle_file
from huggingface_hub import get_token


WAN22_SPACE = "zerogpu-aoti/wan2-2-fp8da-aoti-faster"
WAN22_API_NAME = "/generate_video"
MAX_GENERATIONS_PER_24_HOURS = 3


class Wan22ConfigurationError(RuntimeError):
    pass


class Wan22QuotaProtected(RuntimeError):
    pass


class Wan22CloudClient:
    """Free Hugging Face ZeroGPU client for Wan2.2 Image-to-Video."""

    provider_name = "Hugging Face ZeroGPU Wan2.2"
    artifact_name = "wan22"

    def __init__(self, log: Callable[[str], None] | None = None):
        token = get_token()
        if not token:
            raise Wan22ConfigurationError(
                "Hugging Face 로그인이 필요합니다. `hf auth login`을 먼저 실행하세요."
            )
        self.log = log or (lambda _message: None)
        self.token = token
        self.base_dir = Path(__file__).resolve().parents[1]
        self.cache_dir = self.base_dir / "outputs" / "wan22_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, image_path: Path, prompt: str, seed: int) -> str:
        digest = hashlib.sha256()
        digest.update(image_path.read_bytes())
        digest.update(prompt.encode("utf-8"))
        digest.update(f"{seed};steps=4;duration=2.0".encode("ascii"))
        return digest.hexdigest()[:24]

    def _recent_generations(self) -> list[Path]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        candidates = list((self.base_dir / "outputs").glob("*/wan22/*.mp4"))
        candidates.extend(self.cache_dir.glob("*.mp4"))
        recent: list[Path] = []
        seen: set[tuple[int, int]] = set()
        for path in candidates:
            try:
                stat = path.stat()
                modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            except OSError:
                continue
            identity = (stat.st_size, int(stat.st_mtime))
            if modified >= cutoff and identity not in seen:
                seen.add(identity)
                recent.append(path)
        return recent

    def _fallback_video(self, preferred_name: str | None = None) -> Path | None:
        candidates = list(self.cache_dir.glob("*.mp4"))
        candidates.extend((self.base_dir / "outputs").glob("*/wan22/*.mp4"))
        demo = self.base_dir / "static" / "media" / "svd-first-day.mp4"
        if demo.is_file():
            candidates.append(demo)
        existing = [path for path in candidates if path.is_file()]
        if preferred_name:
            matching = [path for path in existing if path.name == preferred_name]
            if matching:
                return max(matching, key=lambda path: path.stat().st_mtime)
        return max(existing, key=lambda path: path.stat().st_mtime) if existing else None

    def generate(
        self,
        image_path: Path,
        motion_prompt: str,
        output_path: Path,
        *,
        seed: int,
    ) -> dict:
        if not image_path.is_file():
            raise FileNotFoundError(f"Wan2.2 입력 이미지를 찾을 수 없습니다: {image_path}")

        cache_path = self.cache_dir / f"{self._cache_key(image_path, motion_prompt, seed)}.mp4"
        if cache_path.is_file():
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache_path, output_path)
            self.log(f"Wan2.2 캐시 사용: {cache_path.name}")
            return {
                "provider": "local Wan2.2 cache",
                "space": WAN22_SPACE,
                "seed": seed,
                "prompt": motion_prompt,
                "cached": True,
                "quota_fallback": False,
            }

        recent = self._recent_generations()
        if len(recent) >= MAX_GENERATIONS_PER_24_HOURS:
            raise Wan22QuotaProtected(
                f"무료 사용 보호: 최근 24시간 {len(recent)}회 생성. "
                "프롬프트 이미지에 로컬 모션을 적용합니다."
            )

        self.log(f"HF ZeroGPU Wan2.2 대기열 등록: {image_path.name}")
        client = Client(WAN22_SPACE, token=self.token, verbose=False)
        result = client.predict(
            input_image=handle_file(str(image_path)),
            prompt=motion_prompt,
            steps=4,
            negative_prompt=(
                "flicker, frozen frame, distorted face, deformed hands, duplicate person, "
                "inconsistent character, text, logo, watermark, low quality"
            ),
            # The renderer loops this short clip to match the scene duration.
            duration_seconds=2.0,
            guidance_scale=1.0,
            guidance_scale_2=1.0,
            seed=seed,
            randomize_seed=False,
            api_name=WAN22_API_NAME,
        )

        video_result, returned_seed = result
        source = Path(video_result["video"] if isinstance(video_result, dict) else video_result)
        if not source.is_file():
            raise RuntimeError(f"Wan2.2 Space가 유효한 MP4를 반환하지 않았습니다: {result}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, output_path)
        shutil.copy2(source, cache_path)
        return {
            "provider": "Hugging Face ZeroGPU",
            "space": WAN22_SPACE,
            "seed": returned_seed,
            "prompt": motion_prompt,
            "cached": False,
            "quota_fallback": False,
        }

    def close(self) -> None:
        """Match the local provider lifecycle interface."""
