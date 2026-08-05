from __future__ import annotations

import json
import os
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont, ImageOps

from studio.wan22_cloud import Wan22CloudClient


WIDTH = 720
HEIGHT = 1280
FPS = 24


class VideoRenderer:
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.output_root = base_dir / "outputs"
        self.demo_assets = base_dir / "assets" / "demo"
        self.tts_script = base_dir / "scripts" / "tts.ps1"
        self.ffmpeg = self._find_binary("ffmpeg.exe")
        self.ffprobe = self.ffmpeg.with_name("ffprobe.exe")
        self.font_regular = self._find_font(
            "malgun.ttf", "malgunsl.ttf", "gulim.ttc", "batang.ttc"
        )
        self.font_bold = self._find_font(
            "malgunbd.ttf", "malgun.ttf", "gulim.ttc", "batang.ttc"
        )

    @staticmethod
    def _find_binary(name: str) -> Path:
        found = shutil.which(name) or shutil.which(name.removesuffix(".exe"))
        if found:
            return Path(found)

        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            matches = sorted(root.glob(f"Gyan.FFmpeg_*/ffmpeg-*/bin/{name}"), reverse=True)
            if matches:
                return matches[0]
        raise RuntimeError("FFmpeg를 찾을 수 없습니다. winget install Gyan.FFmpeg를 실행하세요.")

    @staticmethod
    def _find_font(*names: str) -> Path:
        fonts_dir = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        for name in names:
            font = fonts_dir / name
            if font.is_file():
                return font
        searched = ", ".join(names)
        raise RuntimeError(
            "사용 가능한 한글 폰트를 찾을 수 없습니다. "
            f"Windows 한국어 추가 글꼴을 설치하세요. 검색 위치: {fonts_dir} "
            f"(파일: {searched})"
        )

    def render(self, project_id: str, story: dict) -> dict:
        project_dir = self.output_root / project_id
        scenes_dir = project_dir / "scenes"
        audio_dir = project_dir / "audio"
        clips_dir = project_dir / "clips"
        cloud_dir = project_dir / "wan22"
        overlay_dir = project_dir / "overlays"
        for directory in (scenes_dir, audio_dir, clips_dir, cloud_dir, overlay_dir):
            directory.mkdir(parents=True, exist_ok=True)

        # Keep all nine story beats, but spend ZeroGPU time on only three hero
        # shots. The other beats use distinct stills with local camera motion.
        # This preserves the 40-second shared story and both 20-second endings
        # without looping one Wan clip for an entire section.
        common_scenes = story["common_scenes"]
        ending_a_scenes = story["ending_a"]
        ending_b_scenes = story["ending_b"]
        all_scenes = common_scenes + ending_a_scenes + ending_b_scenes
        wan_scene_ids = {
            common_scenes[0]["id"],
            ending_a_scenes[-1]["id"],
            ending_b_scenes[-1]["id"],
        }
        clip_paths: dict[str, Path] = {}
        cloud_client = Wan22CloudClient()
        cloud_jobs: dict[str, dict] = {}

        for index, scene in enumerate(all_scenes, start=1):
            render_duration = scene["duration"]
            image_path = scenes_dir / f"{scene['id']}.png"
            text_path = audio_dir / f"{scene['id']}.txt"
            audio_path = audio_dir / f"{scene['id']}.wav"
            raw_video_path = cloud_dir / f"{scene['id']}.mp4"
            overlay_path = overlay_dir / f"{scene['id']}.png"
            clip_path = clips_dir / f"{scene['id']}.mp4"

            self._draw_scene(image_path, story, scene, index, len(all_scenes))
            self._draw_subtitle_overlay(
                overlay_path, story, scene, index, len(all_scenes)
            )
            text_path.write_text(scene["dialogue"], encoding="utf-8")
            self._synthesize(text_path, audio_path, scene["speaker"])
            if scene["id"] in wan_scene_ids:
                input_image = self._input_image_for_scene(scene)
                generation_prompt = self._generation_prompt(story, scene)
                cloud_jobs[scene["id"]] = cloud_client.generate(
                    input_image,
                    generation_prompt,
                    raw_video_path,
                    seed=2100 + index,
                )
                cloud_jobs[scene["id"]]["render_duration"] = render_duration
                self._render_cloud_clip(
                    raw_video_path,
                    audio_path,
                    overlay_path,
                    clip_path,
                    render_duration,
                )
            else:
                cloud_jobs[scene["id"]] = {
                    "provider": "local animated still",
                    "cached": True,
                    "render_duration": render_duration,
                }
                self._render_clip(image_path, audio_path, clip_path, render_duration)
            clip_paths[scene["id"]] = clip_path

        common = project_dir / "common.mp4"
        ending_a = project_dir / "ending_a.mp4"
        ending_b = project_dir / "ending_b.mp4"
        full_a = project_dir / "full_ending_a.mp4"
        full_b = project_dir / "full_ending_b.mp4"

        self._concat([clip_paths[s["id"]] for s in common_scenes], common)
        self._concat([clip_paths[s["id"]] for s in ending_a_scenes], ending_a)
        self._concat([clip_paths[s["id"]] for s in ending_b_scenes], ending_b)
        self._concat([common, ending_a], full_a)
        self._concat([common, ending_b], full_b)

        (project_dir / "story.json").write_text(
            json.dumps(story, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (project_dir / "wan22_jobs.json").write_text(
            json.dumps(cloud_jobs, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "poster": "scenes/common_01.png",
            "videos": {
                "common": common.name,
                "ending_a": ending_a.name,
                "ending_b": ending_b.name,
                "full_a": full_a.name,
                "full_b": full_b.name,
            },
        }
        (project_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    @staticmethod
    def _generation_prompt(story: dict, scene: dict) -> str:
        """Build a prompt grounded in the user's premise and this exact scene.

        Wan2.2 is image-to-video, so the reference image controls identity and
        composition while this prompt controls the scene-specific action and mood.
        Keeping the original user premise in every request prevents unrelated
        prompts from producing the same generic office motion.
        """
        return "\n".join(
            (
                "Create one coherent shot for a Korean short-form drama.",
                f"User story premise: {story['logline']}",
                f"Genre: {story['genre']}. Mood: {story['mood']}.",
                f"Scene title: {scene['title']}.",
                f"Scene content: {scene['visual_prompt']}",
                f"Spoken line and emotional context: {scene['dialogue']}",
                f"Required motion: {scene['motion_prompt']}",
                (
                    "Preserve the people, clothing, location, framing, and visual "
                    "identity from the input image. Animate only actions that fit "
                    "this scene. Natural restrained movement, consistent faces, "
                    "cinematic realism, no text or subtitles in the generated video."
                ),
            )
        )

    def _input_image_for_scene(self, scene: dict) -> Path:
        configured = scene.get("image_path")
        if configured:
            candidate = Path(configured)
            if not candidate.is_absolute():
                candidate = self.base_dir / candidate
            if not candidate.is_file():
                raise RuntimeError(f"장면 입력 이미지를 찾을 수 없습니다: {candidate}")
            return candidate
        return self._background_for_scene(scene["id"])

    def _font(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(str(self.font_bold if bold else self.font_regular), size)

    @staticmethod
    def _hex(hex_color: str) -> tuple[int, int, int]:
        value = hex_color.lstrip("#")
        return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))

    def _draw_scene(
        self,
        output: Path,
        story: dict,
        scene: dict,
        index: int,
        total: int,
    ) -> None:
        accent = self._hex(scene["accent"])
        background = self._background_for_scene(scene["id"])
        image = ImageOps.fit(
            Image.open(background).convert("RGB"),
            (WIDTH, HEIGHT),
            method=Image.Resampling.LANCZOS,
        )

        if scene["phase"] == "ending_a":
            image = ImageEnhance.Color(image).enhance(0.72)
        elif scene["phase"] == "ending_b":
            warm = Image.new("RGB", image.size, (255, 137, 54))
            image = Image.blend(image, warm, 0.08)

        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay, "RGBA")
        overlay_draw.rectangle((0, 0, WIDTH, 170), fill=(3, 5, 10, 105))
        for y in range(690, HEIGHT):
            opacity = int(210 * ((y - 690) / (HEIGHT - 690)) ** 0.72)
            overlay_draw.line((0, y, WIDTH, y), fill=(3, 4, 9, opacity))
        image = Image.alpha_composite(image.convert("RGBA"), overlay)
        draw = ImageDraw.Draw(image, "RGBA")

        draw.rounded_rectangle((34, 34, 686, 104), 24, fill=(5, 7, 12, 155))
        draw.text((58, 53), f"SCENE {index:02}  •  {scene['title']}", font=self._font(20, True), fill=(255, 255, 255))
        draw.text((620, 53), f"{index:02}", font=self._font(20, True), fill=(*accent, 255))

        draw.rounded_rectangle((42, 887, 232, 940), 18, fill=(*accent, 235))
        speaker = "NARRATOR" if scene["speaker"] == "내레이션" else scene["speaker"]
        draw.text((65, 900), speaker, font=self._font(21, True), fill=(255, 255, 255))

        subtitle_lines = textwrap.wrap(scene["subtitle"], width=19)
        subtitle_y = 972
        for line in subtitle_lines[:4]:
            draw.text(
                (44, subtitle_y), line, font=self._font(36, True),
                fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 180),
            )
            subtitle_y += 54

        draw.text((44, 1230), story["title"], font=self._font(17), fill=(220, 222, 230, 190))
        image.convert("RGB").save(output, quality=95)

    def _background_for_scene(self, scene_id: str) -> Path:
        if scene_id in {"common_01", "common_02"}:
            filename = "scene-01-first-day.png"
        elif scene_id in {"common_03", "common_04"}:
            filename = "scene-02-reveal.png"
        else:
            filename = "scene-03-confrontation.png"

        path = self.demo_assets / filename
        if not path.exists():
            raise RuntimeError(f"드라마 장면 이미지를 찾을 수 없습니다: {path}")
        return path

    def _draw_subtitle_overlay(
        self,
        output: Path,
        story: dict,
        scene: dict,
        index: int,
        total: int,
    ) -> None:
        accent = self._hex(scene["accent"])
        overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")
        draw.rounded_rectangle((34, 34, 686, 104), 24, fill=(5, 7, 12, 150))
        draw.text(
            (58, 53), f"SCENE {index:02}  •  {scene['title']}",
            font=self._font(20, True), fill=(255, 255, 255),
        )
        draw.text((620, 53), f"{index:02}", font=self._font(20, True), fill=(*accent, 255))

        for y in range(750, HEIGHT):
            opacity = int(215 * ((y - 750) / (HEIGHT - 750)) ** 0.7)
            draw.line((0, y, WIDTH, y), fill=(3, 4, 9, opacity))

        draw.rounded_rectangle((42, 887, 232, 940), 18, fill=(*accent, 235))
        speaker = "NARRATOR" if scene["speaker"] == "내레이션" else scene["speaker"]
        draw.text((65, 900), speaker, font=self._font(21, True), fill=(255, 255, 255))

        subtitle_y = 972
        for line in textwrap.wrap(scene["subtitle"], width=19)[:4]:
            draw.text(
                (44, subtitle_y), line, font=self._font(36, True),
                fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0, 180),
            )
            subtitle_y += 54
        draw.text((44, 1230), story["title"], font=self._font(17), fill=(220, 222, 230, 190))
        overlay.save(output)

    def _synthesize(self, text_path: Path, output: Path, speaker: str) -> None:
        rate = -1 if speaker == "내레이션" else 0
        command = [
            "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(self.tts_script),
            "-TextPath", str(text_path),
            "-OutputPath", str(output),
            "-VoiceName", "Microsoft Heami Desktop",
            "-Rate", str(rate),
        ]
        self._run(command)

    def _render_clip(self, image: Path, audio: Path, output: Path, duration: int) -> None:
        frames = duration * FPS
        video_filter = (
            f"zoompan=z='min(zoom+0.0008,1.08)':"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
            f"d={frames}:s={WIDTH}x{HEIGHT}:fps={FPS},format=yuv420p"
        )
        command = [
            str(self.ffmpeg), "-y", "-loop", "1", "-i", str(image),
            "-i", str(audio), "-vf", video_filter,
            "-af", f"apad=pad_dur={duration}", "-t", str(duration),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(output),
        ]
        self._run(command)

    def _render_cloud_clip(
        self,
        video: Path,
        audio: Path,
        overlay: Path,
        output: Path,
        duration: int,
    ) -> None:
        video_filter = (
            f"[0:v]scale=800:1422:force_original_aspect_ratio=increase,"
            f"crop={WIDTH}:{HEIGHT}:"
            f"x='(iw-ow)/2+20*sin(t*0.35)':"
            f"y='(ih-oh)/2+24*sin(t*0.22)'[scene];"
            f"[scene][2:v]overlay=0:0:format=auto,format=yuv420p[outv]"
        )
        command = [
            str(self.ffmpeg), "-y",
            "-stream_loop", "-1", "-i", str(video),
            "-i", str(audio),
            "-loop", "1", "-i", str(overlay),
            "-filter_complex", video_filter,
            "-map", "[outv]", "-map", "1:a",
            "-af", f"apad=pad_dur={duration}", "-t", str(duration),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
            str(output),
        ]
        self._run(command)

    def _concat(self, inputs: list[Path], output: Path) -> None:
        list_file = output.with_suffix(".concat.txt")
        lines = [f"file '{path.as_posix().replace(chr(39), chr(39) * 2)}'" for path in inputs]
        list_file.write_text("\n".join(lines), encoding="utf-8")
        command = [
            str(self.ffmpeg), "-y", "-f", "concat", "-safe", "0",
            "-i", str(list_file), "-c", "copy", "-movflags", "+faststart",
            str(output),
        ]
        self._run(command)

    @staticmethod
    def _run(command: list[str]) -> None:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-12:]
            raise RuntimeError("외부 프로그램 실행 실패:\n" + "\n".join(detail))
