from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import pipeline


# 1. 입력 스키마 정의 (Request Body)
class AnalysisRequest(BaseModel):
    # Field를 사용하면 메타데이터와 제약 조건을 동시에 부여할 수 있습니다.
    text: str = Field(
        ...,  # 필수 값임을 의미
        min_length=2,  # 최소 길이 제약
        max_length=500,  # 최대 길이 제약
        description="분석할 리뷰 텍스트입니다.",  # swagger 문서에 표시될 설명
    )

    user_id: Optional[str] = Field(None, example="user_123")  # 선택적 필드 예시


# 2. 출력 스키마 정의 (Response Body)
class AnalysisResponse(BaseModel):
    label: str  # 예: "POSITIVE" or "NEGATIVE"
    ...  # label 필드는 필수

    confidence: float = Field(..., ge=0, le=1)  # 0과 1 사이의 값 보장
    status: str = "success"  # 기본값 설정


# 3. 모델 로딩 (Lifespan 전략 활용)
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("AI 모델 로딩 중...")

    device = 0 if torch.cuda.is_available() else -1

    app.state.classifier = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device,
    )
    yield

    print("자원 해제 중...")
    del app.state.classifier
    print("자원 해제 완료.")


app = FastAPI(lifespan=lifespan, title="감정 분류 서버")


@app.post(
    "/analyze",  # 엔드포인트 경로
    response_model=AnalysisResponse,  # 출력 스키마 지정
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

        # 출력 스키마에 맞춰 객체 생성 및 반환
        return AnalysisResponse(
            label=model_result["label"], confidence=model_result["score"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model Inference Error: {str(e)}")


# uvicorn 03_pydatic_serving:app --reload

"""
테스트 문장 : 
- 한국어 : 이 영화 정말 최고예요! 강력 추천합니다.
- 영어 : This movie is fantastic! Highly recommended.

- 한국어 : 이 영화 정말 최악이에요. 시간 낭비였어요.
- 영어 : This movie is terrible. It was a waste of time.

실패 케이스
- a
"""

# 실행 결과 코드 분석:
# 만약 사용자가 "A" (한 글자)를 보내면, FastAPI는 422 Unprocessable Entity 에러를
# 자동으로 반환하며 "텍스트는 최소 2자 이상이어야 합니다"라는 메시지를 포함합니다.
