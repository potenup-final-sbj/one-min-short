from contextlib import asynccontextmanager
import torch
from fastapi import FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator, ValidationError
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
    text: str = Field(..., description="분석할 리뷰 텍스트")

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
@app.post("/analyze-local")
# data: Dict[str, Any]) : AnalysisRequest를 사용하는 경우 rust에서 exception을 처리한다.
# 이로 인해 exception block이 동작하지 않으며 결국 FastAPI의 기본 예외 처리기로 넘어가게 됩니다.
# 별도의 예외 처리를 위해서는 dict 형태로 수신한 후 수동 검증을 수행해야 합니다.
async def analyze_sentiment_local(data: Dict[str, str]): 
    """
    '자동 검증' 대신 '수동 검증'을 통해 
    이 API 전용 로컬 예외 처리를 수행합니다.
    """
    try:
        # 1. 수동 검증 수행 (여기서 AnalysisRequest의 @field_validator가 작동함)
        # model_validate : pydantic에서 제공하는 메서드로 dict 데이터를 모델 인스턴스로 변환하며 검증을 수행합니다.
        request_data = AnalysisRequest.model_validate(data)
        
        # 2. 모델 추론 수행
        model_result = app.state.classifier(request_data.text)[0]

        # 3. 결과 반환 (결과값 중 label만 추출하도록 수정)
        return {
            "status": "success", 
            "data": f"분석 결과: {model_result['label']} (신뢰도: {model_result['score']:.2f})"
        }

    # 4. 예외 처리
    # Dict 형태로 수신한 데이터가 AnalysisRequest 스키마에 맞지 않을 경우
    except ValidationError as e:
        # Pydantic 에러 메시지 중 커스텀 에러 내용만 추출
        raw_errors = e.errors() # errors() : Pydantic ValidationError 객체에서 에러 리스트를 추출하는 메서드
        custom_msg = raw_errors[0]['msg'].replace("Value error, ", "")
        
        # 400 Bad Request 응답 반환
        # HTTPException : FastAPI에서 HTTP 예외를 발생시키는 클래스
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            
            detail={
                "error_code": "INVALID_INPUT",
                "message": custom_msg
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"서버 내부 오류: {str(e)}")
    
    

"""
실행 방법:
uvicorn 02_verification_custom:app --reload --workers 1

테스트 케이스
1) 정상 요청
- 한국어 : 재미있는 fastapi 수업입니다.
- 영어 : This is an interesting FastAPI class.
"""
