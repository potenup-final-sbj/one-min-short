import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from transformers import pipeline

# 1. 모델 저장소 정의
# 실제 모델은 서버 사양에 맞춰 선택합니다.
# 여기서는 간단한 감정 분석 모델을 사용합니다.
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"


# 2. Lifespan 설계
# @asynccontextmanager는 이 함수를 비동기 컨텍스트 매니저로 만듭니다.
# 즉, 서버의 시작과 종료 시점에 특정 코드를 실행할 수 있게 합니다.
# 서버가 시작될 때 모델을 로드하고, 로드가 종료되면
# `yield`를 만나게 되며 이를 통해 서버가 요청을 받기 시작합니다.
# 이후 서버가 종료될 때는 `yield` 이후의 코드가 실행되면서 자원을 정리합니다.
@asynccontextmanager
async def lifespan(app: FastAPI):
    # [STARTUP] 서버가 시작될 때 실행
    print(f"--- [STARTUP] 모델 '{MODEL_NAME}' 로딩 시작 ---")
    start_time = time.time()

    device = 0 if torch.cuda.is_available() else -1
    # 모델을 로드하여 app.state에 저장합니다.
    # app.state는 FastAPI 애플리케이션 인스턴스에 속하는 속성으로, 동적으로 속성을 추가할 수 있습니다.
    app.state.model = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)

    end_time = time.time()
    print(f"--- [STARTUP] 로딩 완료 (소요시간: {end_time - start_time:.2f}s) ---")

    yield  # 여기서 서버가 요청을 받기 시작합니다.

    # [SHUTDOWN] 서버가 종료될 때 실행
    print("--- [SHUTDOWN] 자원 정리 중 ---")
    # del : 파이썬에서 객체를 명시적으로 삭제하는 키워드입니다.
    del app.state.model  # 메모리에서 명시적으로 제거 (Garbage Collection 유도)


# 3. FastAPI 앱 인스턴스 생성 시 lifespan 연결
# lifespan : FastAPI 애플리케이션의 시작과 종료 시점에 실행되는 이벤트 훅을 정의합니다.
# 이를 활용해 서버가 시작될 때 무거운 AI 모델을 메모리에 로드하고, 종료 시점에 자원을 해제할 수 있습니다.
# lifespan의 자원이 준비되기 전까지는 FastAPI가 외부 요청을 받지 않으므로,
# "완벽하게 준비된 상태에서만 장사를 시작하겠다"는 선언이 됩니다.
app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return {"message": "AI Inference Server is Running"}


@app.post("/predict")
async def predict(text: str):
    # app.state에 저장된 모델을 사용하여 'Warm Start' 추론 수행
    # 이미 로드된 상태이므로 매우 빠릅니다.
    result = app.state.model(text)
    return {"result": result}


# 실행 방법:
# uvicorn 01_model-loading:app --reload --workers 1
# --workers 옵션을 통해 프로세스 수를 조절할 수 있습니다.
# 실행 시 모델의 로딩으로 인해 느리게 시작되는 것을 확인할 수 있습니다.

"""
모델 테스트 예시:

# 오늘 날씨가 참 좋네요!
# The weather is so nice today!

# 이 영화 정말 재미없어요.
# This movie is really boring.
"""
