# AI 숏드라마 스튜디오

한 줄 소재를 입력하면 등장인물·대본·장면을 만들고, 세로형 숏드라마 MP4와 A/B 결말을 생성하는 FastAPI 웹 앱입니다.

## 영상 생성 경로

- 모델: Wan2.2 14B Image-to-Video
- 실행: Hugging Face ZeroGPU 공개 Space API
- Space: `zerogpu-aoti/wan2-2-fp8da-aoti-faster`
- 입력: 장면 이미지 + 장면별 영문 동작 프롬프트
- 출력: 장면별 세로형 MP4
- 후처리: Windows SAPI 또는 Linux Edge TTS, 자막, FFmpeg 합성

fal.ai와 결제 API는 사용하지 않습니다. Hugging Face 무료 계정은 ZeroGPU를 하루 5분까지 사용할 수 있으며 대기열이 생길 수 있습니다. 무료 할당량에 맞추기 위해 각 장면은 2초짜리 원본 모션을 만들고, FFmpeg가 대사 길이에 맞춰 반복합니다.

SVD 모델 캐시와 `scripts/generate_motion_clip.py`는 예비용으로만 보관하며 운영 파이프라인에서는 호출하지 않습니다.

## 준비

Hugging Face 로그인이 필요합니다.

```powershell
hf auth login
```

이미 로그인했다면 다시 로그인할 필요가 없습니다. 토큰을 브라우저 코드나 프론트엔드에 넣지 않습니다.

### Ubuntu/Debian 준비

```bash
sudo apt update
sudo apt install -y ffmpeg fonts-noto-cjk git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Linux 음성은 `edge-tts`, 한글 자막은 Noto CJK 폰트를 사용합니다. Python 의존성은 `uv sync`가 설치합니다.

## 실행

```powershell
cd C:\Users\user\Desktop\16_hugginface\03_video
uv sync
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8030
```

브라우저에서 <http://127.0.0.1:8030>을 엽니다.

### Linux 실행

```bash
git clone https://github.com/Parkseammul/one-min-short.git
cd one-min-short/03_video
uv sync
uv run hf auth login
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8030
```

같은 PC에서는 <http://127.0.0.1:8030>으로 접속합니다. 다른 PC에서 접속하려면 Linux 방화벽에서 8030 포트를 허용하고 서버 IP를 사용하세요.

## 주요 파일

```text
03_video/
├─ main.py                       # FastAPI 및 생성 API
├─ studio/story_engine.py        # 대본, 장면, A/B 결말
├─ studio/video_renderer.py      # 영상·TTS·자막 합성
├─ studio/wan22_cloud.py         # 무료 HF ZeroGPU Wan2.2 클라이언트
├─ scripts/tts.ps1               # Windows 한국어 TTS
├─ static/                       # 웹 화면
└─ outputs/                      # 생성 결과(깃 제외)
```
