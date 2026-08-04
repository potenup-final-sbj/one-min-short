from fastapi import FastAPI
from contextlib import asynccontextmanager
from transformers import pipeline
import anyio # FastAPI 설치 시 함께 설치되는 비동기 유틸리티
import time

"""
ayyio 란?
anyio는 Python에서 비동기 프로그래밍을 쉽게 할 수 있도록 도와주는 라이브러리입니다.
특히, asyncio와 trio라는 두 가지 주요 비동기 프레임워크를 추상화하여, 
개발자가 특정 프레임워크에 종속되지 않고 코드를 작성할 수 있게 해줍니다.
이를 통해 다양한 비동기 환경에서 일관된 방식으로 작업을 처리할 수 있습니다

주로 다음과 같은 기능을 제공합니다:
1. 작업 스케줄링: 비동기 작업을 효율적으로 스케줄링하고 실행할 수 있습니다.
2. 스레드 및 프로세스 관리: 동기 함수를 비동기 코드에서 실행할 수 있도록 도와줍니다.
3. 동시성 제어: 세마포어, 이벤트 등 동시성 제어 메커니즘을 제공합니다.
4. 타임아웃 및 취소: 비동기 작업에 대한 타임아웃 설정과 취소 기능을 지원합니다.
5. 호환성: asyncio와 trio 모두에서 작동하는 코드를 작성할 수 있게 해줍니다.
"""

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 모델 로드
    app.state.model = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    yield

app = FastAPI(lifespan=lifespan)

# 핵심 함수: 동기적인 모델 추론을 감싸는 래퍼
def sync_model_inference(model, text: str):
    # 실제 실무 로직: 무거운 연산을 여기서 수행
    time.sleep(5) # 무거운 연산 가정
    return model(text)

"""
워커 스레드에서 별도의 스레드가 작업을 수행할 수 있도록 하기 위해서
GIL(Global Interpreter Lock)을 해제하는 anyio.to_thread.run_sync를 사용합니다.
이는 파이썬의 객체를 직접 참조하는 것이 아닌 메모리 주소를 직접 가져와 값을 읽어오기 때문에
참조 카운트를 증가시키지 않아 GIL의 영향을 받지 않도록 설계되어 있습니다.

그러나 내부에서 파이썬 객체(외부 변수)를 조작하는 경우에는 GIL의 영향을 받을 수 있으므로 주의가 필요합니다.

"""
@app.get("/fast-inference")
async def fast_inference(text: str):
    print("--- 추론 요청 수신 ---")
    
    # to_thread.run_sync : anyio의 to_thread.run_sync 함수를 사용하여
    # sync_model_inference 함수를 별도의 스레드에서 실행합니다.
    # 이렇게 하면 메인 이벤트 루프가 차단되지 않고 다른 요청을 처리할 수 있습니다.
    predict = anyio.to_thread.run_sync(
        sync_model_inference,   # 별도의 스레드에서 실행할 동기 함수
        app.state.model,        # 모델 객체        
        text,                   # 함수 인자
    )
    
    result = await predict
    print("--- 추론 응답 완료 ---")
    return {"result": result}


"""
FastAPI는 `def`로 정의된 동기 함수는 기본적으로 별도의 스레드에서 실행되도록 설계되어 있습니다.
따라서 이벤트 루프를 블로킹하지 않고 단지 할당된 특정 '워커 스레드' 하나를 5초 동안 점유할 뿐입니다.

그러나 `def`로 정의된 동기 함수의 경우 cpu-bound 작업이 끝나기 전까지는 해당 워커 스레드가 다른 요청을 처리할 수 없게 됩니다.
반대로 `async def`로 정의된 비동기 함수는 cpu-bound 작업을 워커 스레드로 보내고 다른 작업을 수행한 뒤
필요한 시점에 결과를 await하여 받아올 수 있습니다.
"""
@app.get("/blocking")
def blocking():
    # 이 API는 동기적으로 동작하며, 호출 시 서버가 블로킹됩니다.
    print("--- 블로킹 요청 수신 ---")
    time.sleep(5)  # 5초 동안 블로킹
    print("--- 블로킹 응답 완료 ---")
    return {"message": "This was a blocking request."}



@app.get("/ping")
async def ping():
    # 이제 /fast-inference가 동작 중이어도 이 API는 즉시 응답합니다!
    return {"message": "pong"}