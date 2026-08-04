"""
미션 1: 상세 설정 필드 추가 (Config)
요구사항:
1. AnalysisRequest 모델에 include_score: bool = True 필드를 추가하세요.
2. 만약 include_score가 False라면, 응답에서 confidence 점수를 0.0으로 가려버리는 로직을 엔드포인트에 구현하세요.
3. Pydantic의 example 기능을 사용하여 Swagger UI에 멋진 샘플 데이터를 노출하세요.

힌트:
Field(..., example="샘플 문장")을 활용하세요.
"""


from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator, ValidationError
from contextlib import asynccontextmanager
import torch
from transformers import pipeline
from typing import Any, Dict

# 3. 모델 로딩 (Lifespan 전략 활용)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI 모델 로딩 중...")

    # torch.device 객체나 문자열을 사용하는 것이 가장 안전합니다.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        app.state.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=0 if device == "cuda" else -1, # pipeline은 정수형 입력을 선호함
            framework="pt" # PyTorch 명시
        )
        
    except Exception as e:
        print(f"모델 로딩 실패: {e}")
        raise e
    
    
    yield

    print("--- [SHUTDOWN] 자원 해제 중 ---")
    # hasattr : 객체가 특정 속성을 가지고 있는지 확인하는 내장 함수
    if hasattr(app.state, "classifier"):
        del app.state.classifier
    print("자원 해제 완료.")
    

app = FastAPI(lifespan=lifespan, title="감정 분류 서버")
# [Step 1] Pydantic 모델 정의 (커스텀 검증 및 메시지 격리)
class AnalysisRequest(BaseModel):
    # field_validator를 사용하여 커스텀 검증 로직을 추가합니다.
    # 여기서 Field의 역할은 메타데이터 제공에 국한됩니다.
    text: str = Field(
        ..., 
        description="분석할 리뷰 텍스트",
        example="This is a sample review text for sentiment analysis."
    )

    include_score: bool = Field(
        True,
        description="응답에 신뢰도 점수를 포함할지 여부",
        example=True
    )

    # @validator 데코레이터 대신 field_validator 사용한다.
    # @validator(pydantic v1) :파이썬 기반: -> @field_validator(pydantic v2) :러스트 기반: 변경되었다.
    # @field_validator : 러스트 기반의 pydantic v2에서 제공하는 필드 검증기로 속도가 20배 정도 빠르다
    # 파이썬 객체 생성 시 `text` 필드에 대해 자동으로 호출되는 검증기
    @field_validator("text")
    @classmethod  # 자바의 staticmethod과 유사한 개념
    def check_text(cls, v: str) -> str:
        
        if len(v.strip()) < 5:
            # 이 메시지가 나중에 단일 예외 처리에서 추출될 핵심 메시지입니다.
            raise ValueError(
                "입력하신 문장이 너무 짧습니다. 최소 5자 이상 작성해주세요."
            )
        return v
# AnalysisRequest 요약 : text 필드는 필수이며, 최소 5자 이상 / 미만 시 ValueError 발생

# [Step 2] 단일 엔드포인트 내 예외 처리 (Local Exception Handling)
@app.post(
    "/analyze",  # 엔드포인트 경로
    response_model=Dict[str, Any],  # 출력 스키마 지정
)
async def analyze_sentiment(request: AnalysisRequest):
    """
    텍스트를 분석하여 긍정/부정 결과를 반환합니다.
    """
    # request 객체는 이미 Pydantic에 의해 검증이 완료된 상태입니다.
    # .text 속성으로 안전하게 접근 가능합니다.
    try:
        # 모델 추론 수행
        model_result = app.state.classifier(request.text)[0]

        # include_score 값에 따라 confidence 점수 조정
        confidence_score = model_result["score"] if request.include_score else 0.0

        # 출력 스키마에 맞춰 객체 생성 및 반환
        return {
            "label": model_result["label"],
            "confidence": confidence_score,
            "status": "success"
        }

    except Exception as e:
        print(f"추론 중 오류 발생: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="모델 추론 중 오류가 발생했습니다."
        )