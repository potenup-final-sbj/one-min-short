import torch
from transformers import AutoModel, AutoTokenizer

# 1. 모델과 토크나이저 준비
model_name = "google-bert/bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 서로 길이가 다른 문장들
sentences = ["오늘 날씨가 정말 좋네요.", "AI 기술은 우리의 삶을 변화시키고 있습니다!"]

# 3. 배치 토크나이징
# - padding='longest': 배치 내에서 가장 긴 문장에 맞춰 패딩을 채움
batch_inputs = tokenizer(
    sentences,  # 토크나이징할 문장들
    padding="longest",  # 문장 길이 맞춤
    truncation=True,  # 너무 긴 문장은 자름
    return_tensors="pt",  # 파이토치 텐서로 반환
)

print("--- [배치 처리 내부 구조] ---")
print(f"Input IDs 형태: {batch_inputs['input_ids'].shape}")  # [문장수, 최대길이]
print(f"Input IDs:\n{batch_inputs['input_ids']}")
print(f"Attention Mask:\n{batch_inputs['attention_mask']}")

# 4. 모델 추론 (통역 완료된 데이터를 요리사에게 전달)
# no_grad() : torch의 자동 미분 기능을 비활성화하여 메모리 사용량 최적화
with torch.no_grad():  # torch의 환경 설정을 변경
    # 모델의 추론 수행
    outputs = model(**batch_inputs)  # 배치 입력을 모델에 전달

# 5. 결과 분석
# last_hidden_state : 최종 레이어의 은닉 상태 출력
print(f"\n최종 출력 텐서 형태: {outputs.last_hidden_state.shape}")
# [2(문장수), 11(토큰길이), 768(특성차원)]
# 현재 컴퓨터가 두 문장을 각각 11개의 토큰으로 쪼개어,
# 각 토큰을 768차원의 벡터로 변환했음을 의미합니다


"""
현재의 AutoModel은 입력 값의 대한 이해만 가능합니다.
사람이 입력한 언어를 모델의 토크나이저가 알아 들을 수 있는 숫자 ID로 변환하고,
그 숫자 ID 행렬을 모델에 전달하고 모델은 연산(추론)을 통해 사람의 말을 이해합니다. 
위 과정까지를 인코딩(Encoding)이라고 부릅니다.
하지만 인코딩은 모델이 사람의 말을 이해하는 것에 불과합니다.
사람과 대화를 하기 위해서는 모델이 이해한 내용을 다시 사람이 알아들을 수 있는 언어로 바꿔줘야 합니다.
이 과정을 디코딩(Decoding)이라고 부르며 디코딩은 모델의 task에 따라 별도의 디코더 모델이 필요합니다.
"""
