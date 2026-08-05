from __future__ import annotations

import asyncio
import os
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from studio.story_engine import create_story
from studio.video_renderer import VideoRenderer
from studio.wan22_cloud import WAN22_SPACE
from huggingface_hub import get_token


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

app = FastAPI(title="AI Short Drama Studio", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

renderer = VideoRenderer(BASE_DIR)
render_lock = asyncio.Lock()


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=5, max_length=300)
    genre: str = Field(default="로맨스")
    mood: str = Field(default="긴장감 있는")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "status": "ok",
        "ffmpeg": str(renderer.ffmpeg),
        "tts": "Windows SAPI",
        "video_provider": f"Hugging Face ZeroGPU: {WAN22_SPACE}",
        "wan22_configured": bool(get_token()),
        "billing": "free daily quota",
    }


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> dict:
    prompt = " ".join(request.prompt.split())
    project_id = uuid4().hex[:12]
    story = create_story(prompt, request.genre, request.mood)

    try:
        async with render_lock:
            manifest = await asyncio.to_thread(
                renderer.render,
                project_id,
                story,
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "project_id": project_id,
        "story": story,
        "videos": {
            key: f"/outputs/{project_id}/{value}"
            for key, value in manifest["videos"].items()
        },
        "poster": f"/outputs/{project_id}/{manifest['poster']}",
        "download_name": f"{story['title']}.mp4",
    }
