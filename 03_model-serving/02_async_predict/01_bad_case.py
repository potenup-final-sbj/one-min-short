import asyncio
import time
from fastapi import FastAPI
from transformers import pipeline
from contextlib import asynccontextmanager
import torch



@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    yield



app = FastAPI(lifespan=lifespan)


@app.get("/slow-inference")
async def slow_inference(text: str):
    print("--- 추론 시작 (이벤트 루프 점유) ---")
    # 실제 모델 추론은 CPU 연산이므로 이 동안 루프가 멈춥니다.
    # 인위적인 CPU 부하를 위해 루프를 돌립니다.
    start = time.perf_counter()
    
    # [주의] 이 부분은 CPU-bound 작업을 흉내냅니다.
    # 실제 model(text) 호출과 동일한 효과를 냅니다.
    result = app.state.model(text)
    
    # asyncio.sleep() 대신 time.sleep()을 사용하는 이유는 cpu-bound 작업을 흉내내기 위함입니다.
    time.sleep(5) # 5초 동안 서버 전체가 멈춤
    
    end = time.perf_counter()
    print(f"--- 추론 완료 ({end - start:.2f}s) ---")
    return {"result": result}

@app.get("/ping")
async def ping():
    # 이 API는 즉시 응답해야 하지만, 위 API가 동작 중이면 응답하지 못합니다.
    return {"message": "pong"}


# uvicorn 01_bad_case:app --reload
# 테스트 클라이언트는 01_test_client.py 파일을 참고하세요.
