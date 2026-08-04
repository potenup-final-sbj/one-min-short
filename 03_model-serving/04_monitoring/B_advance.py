from fastapi import FastAPI
from contextlib import asynccontextmanager
from B_model_info import MODEL_INFO
from prometheus_fastapi_instrumentator import Instrumentator

"""
uv add prometheus-fastapi-instrumentator
FastAPI 애플리케이션의 메트릭(요청 수, 응답 시간, 에러율 등)을 Prometheus가 읽을 수 있는 형식으로 자동 생성해주는 라이브러리입니다
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    # settings 객체를 통해 안전하게 설정값에 접근
    print(f"--- [STARTUP] {MODEL_INFO['id']} (v{MODEL_INFO['version']}) 로딩 중 ---")
    
    # 모델 로딩 시 settings.model_base 사용
    # app.state.model = pipeline(..., model=settings.model_base)
    yield
    print("--- [SHUTDOWN] 자원 정리 ---")

app = FastAPI(lifespan=lifespan)

# Prometheus 메트릭 수집기 설정 및 FastAPI 앱에 연결
# Instrumentator 클래스는 FastAPI 애플리케이션의 메트릭을 수집하고 노출하는 기능을 제공합니다.
# instrument(): 모든 HTTP 요청이 들어오고 나갈 때마다 그 과정을 지켜보며 메트릭을 수집합니다.
# expose(app): FastAPI 애플리케이션에 /metrics라는 새로운 엔드포인트(경로)를 생성 
# app.get("/metrics")를 추가하기 위해 app 인스턴스를 전달합니다.
Instrumentator().instrument(app).expose(app)


@app.get("/info")
async def get_model_info():
    return MODEL_INFO
