# AI 숏드라마 스튜디오

한국어 소재, 장르, 분위기를 입력하면 로컬 AI로 대본과 대표 이미지를 만들고, 로컬 LTX 또는 원격 Wan2.2와 FFmpeg를 이용해 30초 세로형 A/B 결말 영상을 생성하는 FastAPI 애플리케이션입니다.

## 현재 사용 모델과 실행 위치

| 단계 | 모델·도구 | 실행 위치 | 외부 크레딧 |
|---|---|---|---|
| 대본 | `Qwen/Qwen3-4B` 4bit NF4 + Hugging Face Transformers | 로컬 GPU | 사용 안 함 |
| 이미지 | `stabilityai/stable-diffusion-xl-base-1.0` + `ByteDance/SDXL-Lightning` 4-step UNet | 로컬 NVIDIA GPU | 사용 안 함 |
| 영상 모션(기본) | `Lightricks/LTX-Video-0.9.5` Image-to-Video | 로컬 NVIDIA GPU | 사용 안 함 |
| 영상 모션(선택) | Wan2.2 14B Image-to-Video, `zerogpu-aoti/wan2-2-fp8da-aoti-faster` | Hugging Face ZeroGPU Space | 무료 할당량 사용 |
| 음성 | Windows SAPI / Linux Edge TTS | 로컬 또는 Edge TTS | HF 크레딧 사용 안 함 |
| 자막·편집 | Pillow + FFmpeg | 로컬 | 사용 안 함 |

Hugging Face Inference Providers의 유료 이미지 API는 호출하지 않습니다. SDXL과 LTX 모델은 최초 실행 때 Hugging Face Hub에서 로컬 캐시로 다운로드하지만, 다운로드 자체는 Inference Providers 크레딧을 차감하지 않습니다. Hugging Face 로그인 토큰은 `VIDEO_PROVIDER=wan22`일 때만 필요합니다.

## 전체 처리 흐름

```text
브라우저에서 소재·장르·분위기 입력
                 │
                 ▼
POST /api/generate ── main.py
                 │
                 ▼
로컬 Qwen/Qwen3-4B ── studio/story_engine.py
  ├─ 제목·로그라인·등장인물·비주얼 바이블
  ├─ 공통 장면 3개 × 6초 = 18초
  └─ A 결말 1개 / B 결말 1개 × 12초
                 │
                 ▼
로컬 SDXL-Lightning ── studio/scene_image_generator.py
  └─ 프롬프트마다 새로운 9:16 이미지 1장 생성
                 │
                 ▼
장면 렌더링 ── studio/video_renderer.py
  ├─ 생성 이미지 1장을 5개 장면에서 공통 사용
  ├─ 대사 TTS 및 자막 오버레이 생성
  ├─ 공통 첫 장면 + A 결말 + B 결말에 선택한 영상 모델 적용(최대 3회)
  └─ 나머지 장면은 로컬 줌·팬 모션 적용
                 │
                 ▼
FFmpeg 합성
  ├─ 공통 18초 + A 결말 12초 = full_ending_a.mp4 (30초)
  └─ 공통 18초 + B 결말 12초 = full_ending_b.mp4 (30초)
```

LTX 또는 Wan2.2가 생성하는 원본 모션은 약 2초이며, 렌더러가 장면 길이에 맞춰 반복합니다. LTX 생성이 실패하면 전체 요청이 실패합니다. Wan2.2가 ZeroGPU 할당량 보호에 걸리면 해당 장면은 생성 이미지에 로컬 줌·팬 모션을 적용해 계속 렌더링합니다.

## 코드 구조

```text
03_video/
├─ main.py
│  ├─ 웹 화면과 outputs 정적 파일 제공
│  ├─ GET /api/health: 모델·FFmpeg·HF 로그인 상태 확인
│  └─ POST /api/generate: 대본 생성 후 전체 렌더링 실행
├─ studio/story_engine.py
│  ├─ Transformers + bitsandbytes로 Qwen3-4B를 4bit NF4 로드·추론
│  ├─ 3개 공통 장면과 A/B 결말 검증
│  └─ 동일 입력 대본 JSON 캐시
├─ studio/scene_image_generator.py
│  ├─ SDXL Base와 Lightning 4-step UNet 로드
│  ├─ 512×896 대표 이미지 1장 생성
│  └─ 요청마다 새 이미지 생성 후 GPU 메모리 해제
├─ studio/wan22_cloud.py
│  ├─ Wan2.2 ZeroGPU Image-to-Video 호출
│  ├─ 동일 이미지·프롬프트·시드 영상 캐시
│  └─ 최근 24시간 최대 3회 호출 보호
├─ studio/ltx_local.py
│  ├─ 로컬 LTX Image-to-Video 실행
│  ├─ CPU offload와 VAE tiling 적용
│  └─ 동일 이미지·프롬프트·시드 영상 캐시
├─ studio/video_provider.py      # VIDEO_PROVIDER 검증 및 공급자 선택
├─ studio/video_renderer.py
│  ├─ 장면 이미지·TTS·자막·클립 생성
│  └─ FFmpeg로 공통/A/B/최종 영상 합성
├─ scripts/tts.ps1            # Windows 한국어 SAPI TTS
├─ static/                    # 브라우저 화면
├─ tests/                     # 이미지 생성기 테스트
└─ outputs/                   # 실행 결과와 캐시(소스 관리 제외)
```

## 캐시 정책

- 대표 이미지: 캐시하지 않습니다. `/api/generate` 요청마다 `generated_images/story_01.png`를 새로 만듭니다.
- 대본: 같은 모델·소재·장르·분위기는 `outputs/story_cache`의 JSON을 재사용합니다.
- LTX 영상: 같은 모델·이미지·프롬프트·시드는 `outputs/ltx_cache`의 MP4를 재사용합니다.
- Wan2.2 영상: 같은 이미지 바이트·모션 프롬프트·시드는 `outputs/wan22_cache`의 MP4를 재사용합니다.
- 서로 다른 프로젝트는 고유 `project_id` 폴더를 사용하므로 이전 사무실 이미지가 새 요청의 이미지로 자동 재사용되지 않습니다.

## 출력 구조

```text
outputs/<project_id>/
├─ generated_images/story_01.png  # 새로 생성한 대표 이미지
├─ scenes/                        # 자막을 포함한 장면 스틸
├─ audio/                         # 대사와 TTS 파일
├─ ltx/ 또는 wan22/               # 선택한 모델의 원본 모션 클립
├─ clips/                         # 장면별 완성 클립
├─ story.json                     # 생성 대본
├─ ltx_jobs.json 또는 wan22_jobs.json # 선택 모델의 생성·캐시·대체 처리 기록
├─ manifest.json
├─ common.mp4                     # 18초 공통부
├─ ending_a.mp4 / ending_b.mp4    # 각 12초 결말
└─ full_ending_a.mp4 / full_ending_b.mp4  # 각 30초 최종본
```

## 준비

필수 조건은 Python 3.12, NVIDIA CUDA GPU, FFmpeg입니다. 현재 설정은 8GB VRAM의 RTX 4070 Laptop GPU에서 SDXL 이미지 생성과 LTX 샘플의 로컬 실행을 확인했습니다. LTX는 CPU offload를 사용하므로 충분한 시스템 RAM과 swap이 필요하며, Qwen3-4B/SDXL/LTX 모델은 최초 실행 때 Hugging Face Hub에서 로컬 캐시로 다운로드되므로 수십 GB의 저장 공간과 충분한 시간이 필요합니다.

```powershell
hf auth login
```

Wan2.2를 선택할 경우에만 `hf auth login`을 추가로 실행합니다.

### Ubuntu/Debian 준비

```bash
sudo apt update
sudo apt install -y ffmpeg fonts-noto-cjk git curl
curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"
```

Linux에서는 음성에 `edge-tts`, 한글 자막에 Noto CJK 글꼴을 사용합니다.

## 실행

### Windows

```powershell
cd C:\Users\user\Desktop\16_hugginface\03_video
uv sync
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8030
```

브라우저에서 <http://127.0.0.1:8030>을 엽니다.

### Linux

```bash
git clone https://github.com/Parkseammul/one-min-short.git
cd one-min-short/03_video
uv sync
uv run hf auth login
uv run python -m uvicorn main:app --host 0.0.0.0 --port 8030
```

다른 PC에서 접속하려면 서버 방화벽에서 8030 포트를 허용하고 `http://<서버-IP>:8030`으로 접속합니다.

## 환경 변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `HF_STORY_MODEL` | `Qwen/Qwen3-4B` | 로컬 대본 생성용 Hugging Face 모델 ID |
| `STORY_MAX_NEW_TOKENS` | `1500` | 대본 생성 최대 출력 토큰 수 |
| `SDXL_BASE_MODEL` | `stabilityai/stable-diffusion-xl-base-1.0` | SDXL 기반 모델 |
| `SDXL_LIGHTNING_MODEL` | `ByteDance/SDXL-Lightning` | Lightning UNet 저장소 |
| `VIDEO_PROVIDER` | `ltx` | 영상 모델 선택: `ltx` 또는 `wan22` |
| `LTX_VIDEO_MODEL` | `Lightricks/LTX-Video-0.9.5` | 로컬 LTX 모델 ID |

### 영상 모델 전환

환경변수는 서버 시작 시 읽습니다. 값을 바꾼 뒤 서버를 다시 시작해야 합니다.

```bash
VIDEO_PROVIDER=ltx uv run python -m uvicorn main:app --host 0.0.0.0 --port 8030
VIDEO_PROVIDER=wan22 uv run python -m uvicorn main:app --host 0.0.0.0 --port 8030
```

## 현재 확인된 사항

- 대본은 Hugging Face Hub에서 `Qwen/Qwen3-4B` 파일을 내려받아 Transformers로 로컬 추론하며 Inference Providers API를 호출하지 않습니다.
- `Qwen/Qwen3-4B` 모델은 bitsandbytes 4bit NF4로 로드하여 약 2~3GB 수준으로 압축하고 CPU 오프로딩을 방지합니다.
- 최대 출력은 1,500토큰입니다. RTX 4070 Laptop 8GB에서 한국어 30초 JSON 대본을 167.3초에 생성했으며, 기존 원본 가중치의 약 10분보다 크게 단축됐습니다.
- SDXL-Lightning 4-step이 512×896 PNG 한 장을 로컬 GPU에서 실제 생성했습니다.
- 로컬 대본·이미지 생성은 Hugging Face Inference Providers 크레딧을 사용하지 않습니다.
- 긴 SDXL 프롬프트는 CLIP의 77토큰 제한으로 뒷부분이 잘릴 수 있으므로 핵심 인물·동물·장소는 `hero_visual_prompt` 앞부분에 배치해야 합니다.
- 한 장의 대표 이미지를 모든 장면에 공유하므로 장면별 구도 변화보다 등장인물 일관성을 우선하는 현재 구조입니다.
