import torch
from transformers import pipeline

"""
링크에서 ffmpeg 바이너리를 다운로드 받아 시스템 경로에 추가해야 합니다.
https://github.com/GyanD/codexffmpeg/releases/tag/2026-01-14-git-6c878f8b82

ffmpeg-2026-01-14-git-6c878f8b82-essentials_build.zip
압축 해제 후, ffmpeg.exe 파일이 있는 경로를 시스템 환경 변수 PATH에 추가하세요.
"""

# 1. 장치 및 모델 설정
# Whisper 모델은 무겁기 때문에 실무에서는 'base'나 'small' 버전을 선호합니다.
device = 0 if torch.cuda.is_available() else -1
stt_engine = pipeline(
    "automatic-speech-recognition",  # 작업 유형 지정
    model="openai/whisper-tiny",  # 매우 빠른 추론을 위해 tiny 버전 사용
    device=device,  # GPU가 있으면 GPU 사용, 없으면 CPU 사용
)

# 2. 음성 데이터 입력 (URL 또는 로컬 파일 경로)
# 허깅페이스 파이프라인은 URL을 주면 자동으로 다운로드 및 리샘플링을 수행합니다.
audio_url = "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"

# 3. 추론 실행
# chunk_length_s: 긴 음성을 몇 초 단위로 쪼개서 처리할지 결정 (VRAM 관리 핵심)
# Whisper 모델은 기본적으로 30초 단위로 쪼개서 처리하는 것을 권장합니다.
# VRAM이란 : Video RAM의 약자로, 그래픽 처리 장치(GPU)에서 사용하는 메모리를 의미합니다.
# cpu 환경에서는 chunk_length_s 설정이 무의미합니다.
result = stt_engine(audio_url, chunk_length_s=30)

print("\n--- [음성 인식 결과] ---")
print(f"인식된 문장: {result['text']}")
