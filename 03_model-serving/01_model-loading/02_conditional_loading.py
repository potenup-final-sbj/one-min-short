import os
import time
from contextlib import asynccontextmanager

import torch
from dotenv import load_dotenv
from fastapi import FastAPI
from transformers import pipeline

"""
uv add python-dotenv

dotenv 파일 생성 (.env)
dotenv는 .env 파일에 저장된 환경 변수를 로드하여
파이썬 애플리케이션에서 사용할 수 있게 해줍니다.
"""

# .env 파일의 변수들을 시스템 환경변수로 로드합니다.
load_dotenv()

# os.getenv(key, default=None)
# `env`에서 key에 해당하는 값을 가져옵니다.
ENV = os.getenv("ENV", "PRODUCTION")

# 1. 환경 변수에 따른 모델 선택
if ENV == "PRODUCTION":
    MODEL_NAME = "siebert/sentiment-roberta-large-english"
else:
    MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"

print(f"Running in {ENV} mode. Loading model: {MODEL_NAME}")


# 2. Lifespan 설계
@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"--- [STARTUP] 모델 '{MODEL_NAME}' 로딩 시작 ---")
    start_time = time.time()

    device = 0 if torch.cuda.is_available() else -1
    app.state.model = pipeline("sentiment-analysis", model=MODEL_NAME, device=device)

    end_time = time.time()
    print(f"--- [STARTUP] 로딩 완료 (소요시간: {end_time - start_time:.2f}s) ---")

    yield

    print("--- [SHUTDOWN] 자원 정리 중 ---")
    del app.state.model


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index():
    return {"message": "AI Inference Server is Running"}


@app.post("/predict")
async def predict(text: str):
    result = app.state.model(text)
    return {"result": result}


@app.get("/info")
def info():
    return {"model_loaded": MODEL_NAME, "environment": ENV}


# 테스트 문장 예시:
# 날씨가 우중충해 보이지만, 마음만은 화창한 하루 되시길 바랍니다!
# The weather looks gloomy, but I hope you have a sunny day at heart!

# 와, 오늘 날씨 참~ 좋다. 비도 오고 바람도 불고 아주 난리네
# Wow, the weather is so nice today. It's raining and windy. It's crazy

# 이번 신곡 비트 진짜 미쳤다...
# The beat for this new song is crazy...

# 오늘 기분이 정말 우울하네
# I'm really feeling down today.

# 실행 방법:
# uvicorn 02_conditional_loading:app --reload --workers 1
