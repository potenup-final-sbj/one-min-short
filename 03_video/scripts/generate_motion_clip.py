from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Animate a drama still with Stable Video Diffusion")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion", type=int, default=110)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")

    source = Image.open(args.input).convert("RGB")
    source = ImageOps.fit(
        source,
        (576, 1024),
        method=Image.Resampling.LANCZOS,
    )

    pipeline = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.enable_model_cpu_offload()
    pipeline.unet.enable_forward_chunking()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    frames = pipeline(
        source,
        width=576,
        height=1024,
        num_frames=25,
        decode_chunk_size=1,
        motion_bucket_id=args.motion,
        noise_aug_strength=0.03,
        generator=generator,
    ).frames[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=7, quality=8)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
