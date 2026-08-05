from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a real motion clip locally")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU를 찾을 수 없습니다.")

    pipeline = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
        pipeline.scheduler.config
    )
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()

    prompt = (
        "cinematic live action Korean romance drama, a Korean woman in her late "
        "twenties wearing a beige business suit walks into a modern Seoul office, "
        "she pauses and looks up in surprise, natural body movement, coworkers moving "
        "softly in the background, realistic video, subtle handheld camera, soft morning light"
    )
    negative = (
        "still image, frozen, illustration, animation, text, watermark, distorted face, "
        "deformed hands, flicker, low quality"
    )
    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    frames = pipeline(
        prompt=prompt,
        negative_prompt=negative,
        num_frames=24,
        height=576,
        width=320,
        num_inference_steps=30,
        guidance_scale=8.5,
        generator=generator,
    ).frames[0]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(args.output), fps=8, quality=8)
    print(args.output.resolve())


if __name__ == "__main__":
    main()

