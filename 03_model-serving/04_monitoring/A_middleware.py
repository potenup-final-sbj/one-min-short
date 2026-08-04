# uv add psutil

import time
import psutil
import torch
from fastapi import FastAPI, Request
from transformers import pipeline
from contextlib import asynccontextmanager

# 모델 버전 정의 (환경 변수 등으로 관리하는 것이 좋습니다)
MODEL_INFO = {
    "id": "sentiment-distilbert-v1",
    "base": "distilbert-base-uncased-finetuned-sst-2-english",
    "version": "1.0.2"
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    print(f"--- [STARTUP] 모델 {MODEL_INFO['id']} 로딩 ---")
    
    device = 0 if torch.cuda.is_available() else -1
    app.state.model = pipeline("sentiment-analysis", model=MODEL_INFO['base'], device=device)
    yield
    
    print("--- [SHUTDOWN] 자원 정리 ---")

app = FastAPI(lifespan=lifespan)


# 1. 모니터링 미들웨어
# middleware는 모든 요청과 응답의 중간에 끼어드는 '검문소' 역할을 합니다.
# 각 요청이 들어올 때마다 시작 시간을 기록하고, 응답이 나갈 때까지의 소요 시간을 계산합니다.
# 자바의 서블릿 필터(Servlet Filter)와 유사한 개념입니다.
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    # Request 객체: 클라이언트로부터 들어온 HTTP 요청에 대한 모든 정보를 담고 있습니다.
    # call_next: 다음 프로세스(핸들러)로 요청을 전달하는 함수입니다.
    
    # 요청 시작 시간 기록 (고해상도 타이머 사용)
    start_time = time.perf_counter()
    
    # 다음 프로세스(핸들러)로 요청 전달
    response = await call_next(request)
    
    # 전체 소요 시간 계산
    process_time = time.perf_counter() - start_time
    
    # 응답 헤더에 성능 및 버전 정보 추가 (클라이언트/프론트엔드 분석용)
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Model-ID"] = MODEL_INFO["id"]
    response.headers["X-Model-Version"] = MODEL_INFO["version"]
    
    return response

@app.post("/predict")
async def predict(text: str):
    # 실제 추론 수행
    result = app.state.model(text)
    return {"data": result}

# pip install prometheus-fastapi-instrumentator를 통해 더 정교한 모니터링도 가능합니다.
# 자동으로 /metrics 엔드포인트를 생성하고, 다양한 메트릭을 수집합니다.
# 위 lib는 Prometheus와 Grafana와 같은 전문 모니터링 툴과 연동할 수 있습니다.
@app.get("/metrics")
async def get_metrics():
    """
    서버 상태를 수치로 반환하는 전용 엔드포인트
    """
    # CPU 및 메모리 사용량 측정
    # psutil : 시스템 및 프로세스 유틸리티 라이브러리
    # .cpu_percent() : CPU 사용량을 백분율로 반환
    cpu_usage = psutil.cpu_percent()
    # .virtual_memory().percent : RAM 사용량을 백분율로 반환
    ram_usage = psutil.virtual_memory().percent
    
    # GPU 정보 (torch 활용)
    gpu_info = "N/A"
    if torch.cuda.is_available():
        gpu_info = {
            # get_device_name() : GPU 이름 반환
            "name": torch.cuda.get_device_name(0),
            # memory_allocated() : 현재 할당된 VRAM 용량 반환 (바이트 단위) 
            "allocated_vram": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB"
        }

    return {
        "system": {
            "cpu": f"{cpu_usage}%",
            "ram": f"{ram_usage}%",
            "gpu": gpu_info
        },
        "model": MODEL_INFO
    }
    
    
"""
테스트 방법:
1. 서버 실행
   uvicorn A_middleware:app --reload --workers 1

2. 예측 요청
- 한국어 : 이 영화 정말 최고예요! 강력 추천합니다.
- 영어 : This movie is fantastic! Highly recommended.

- 한국어 : 이 영화 정말 최악이에요. 시간 낭비였어요.
- 영어 : This movie is terrible. It was a waste of time.
"""