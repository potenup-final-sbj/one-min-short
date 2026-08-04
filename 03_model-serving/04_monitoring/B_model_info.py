from pydantic_settings import BaseSettings, SettingsConfigDict

"""
uv add pydantic-settings
pydantic-settings는 환경 변수 기반 설정 관리를 도와주는 라이브러리입니다.
.env 파일을 자동으로 로드하고, 설정 클래스를 통해 손쉽게 접근할 수 있습니다.
"""

"""
BaseSettings 클래스
: pydantic의 BaseModel을 상속받아 환경 변수로부터 설정 값을 로드하는 기능을 제공합니다.

SettingsConfigDict 클래스
: pydantic v2에서 설정 구성을 정의하는 데 사용되는 클래스입니다.

환경 변수 설정 예시 (.env 파일)
MODEL_ID=sentiment-analysis-pro 

Setting 클래스 정의

"""

class Settings(BaseSettings):
    # 환경 변수명이 클래스 변수명과 같으면 자동으로 값이 주입됩니다.
    # 값이 없을 경우 사용할 기본값(Default)을 설정할 수 있습니다.
    model_id: str = "default-model"
    model_version: str = "1.0.0"
    model_base: str = "distilbert-base-uncased"
    debug_mode: bool = False

    # Pydantic V2 설정 방식
    # model_config는 환경 변수를 .env 파일에서 읽어오도록 지정하는 역할을 합니다.
    # 이후 대소문자는 구분하지 않고 변수 이름을 매칭하여 값을 주입하게 된다.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

# 인스턴스 생성 시점에 .env 파일을 읽어옵니다.
settings = Settings()

# 전역에서 사용할 모델 정보 객체
MODEL_INFO = {
    "id": settings.model_id,
    "version": settings.model_version,
    "base": settings.model_base
}


