"""
실습 목적
- AutoClass의 내부 동작 원리 이해
- 서로 다른 모델 아키텍처를 동일한 코드로 다루는 방법 습득

AutoModel과 AutoConfig 클래스가 없다면 각 모델마다 별도의 클래스를 불러와야 합니다.
예를 들어 BERT 모델을 사용하려면 BertModel을, GPT-2 모델을 사용하려면 GPT2Model을 각각 임포트해야 합니다.
이렇게 되면 코드가 복잡해지고 유지보수가 어려워집니다.
AutoClass는 이러한 문제를 해결하기 위해 도입된 추상화 계층입니다.
"""

from transformers import AutoConfig, AutoModel

"""
AutoConfig : 모델의 설정 파일(config.json)을 불러오는 클래스
AutoModel : 다양한 모델 아키텍처를 자동으로 인스턴스화하는 클래스

pipeline은 내부적으로 AutoModel, AutoTokenizer 등을 사용하여 모델과 토크나이저를 자동으로 로드합니다.
tokenizer와 model을 일일이 불러올 필요 없이, AutoClass 하나로 모든 작업이 끝납니다.
"""


# 테스트할 서로 다른 아키텍처의 모델들
model_names = ["google-bert/bert-base-uncased", "gpt2"]


def analyze_auto_class(model_name):
    print(f"\n>>> 모델 분석: {model_name}")

    # 1. Config만 먼저 가져와서 내부 'model_type' 확인 (메모리 절약)
    # from_pretrained 메서드는 내부적으로 먼저 config.json을 다운로드합니다.
    config = AutoConfig.from_pretrained(model_name)
    print(f"   [Type 확인] 이 모델의 타입은 '{config.model_type}' 입니다.")

    # 2. AutoModel.from_pretrained() : 실제 모델 인스턴스화

    # low_cpu_mem_usage=True는 메모리 점유를 최적화합니다.
    # pytorch에서 모델을 로드할 때 모델을 올릴 수 있도록 메모리에 모델 크기 만큼 공간을 미리 확보하기 위해
    # 임시 데이터를 채우고 실제 모델을 다운받아 채우는 방식으로 동작합니다.
    # 만약 모델이 1GB라면 2GB의 공간을 잠시 사용하게 되는데,
    # low_cpu_mem_usage=True 옵션을 주면 임시 데이터를 채우지 않고 바로 모델을 채우는 방식으로 메모리 사용량을 절약합니다.
    model = AutoModel.from_pretrained(model_name, low_cpu_mem_usage=True)

    # 3. 생성된 객체의 실제 클래스 타입 출력
    print(f"   [Class 확인] 실제 생성된 클래스: {type(model)}")

    # 4. 파라미터 개수 계산 (메모리 규모 파악)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   [규모 확인] 총 파라미터 수: {num_params:,} 개")


for name in model_names:
    analyze_auto_class(name)
