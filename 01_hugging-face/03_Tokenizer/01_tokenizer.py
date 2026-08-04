from transformers import AutoTokenizer

# 1. Factory 패턴을 활용해 토크나이저 로드
# 한국어와 영어를 동시에 지원하는 다국어 모델을 사용해 봅시다.
model_name = "google-bert/bert-base-multilingual-cased"
# AutoTokenizer : 모델 이름만으로 적절한 토크나이저 클래스를 자동으로 선택해 줍니다.
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("================= 토크나이저 정보 =================")
print(tokenizer)
print("===================================================")
"""
주요 내용
1 name_or_path: 모델 이름 또는 경로
2 vocab_size: 어휘 사전 크기 (토큰 개수)
3 special_tokens_map: 특수 토큰 매핑 정보
4. model_max_length: 모델이 처리할 수 있는 최대 문장 길이

토크 토크나이저는 모델 아키텍처에 맞게 설계되어 있습니다.
현재는 nlp 분야에서 가장 널리 사용되는 BERT 계열의 토크나이저가 로드되어 있습니다.
"""


# 2. 테스트 문장 정의
sentence = "AI 엔지니어링은 정말 재밌어요!"

# 3. 토크나이징 수행 (가장 기본적인 형태)
inputs = tokenizer(
    sentence,  # 토크나이징할 문장
    padding=True,  # 문장 길이 맞춤       -> max_length에 맞춰 패딩
    truncation=True,  # 너무 긴 문장은 자름   -> max_length에 맞춰 자름
    max_length=20,  # 최대 길이 설정
    return_tensors="pt",  # 파이토치 텐서로 반환 -> 모델은 텐서 형태의 입력을 받기 때문입니다.
)

print("--- [1. 토크나이징 결과 분석] ---")
print(f"입력 문장: {sentence}")
print(f"Input IDs (숫자): {inputs['input_ids']}")  # ID 행렬 출력
# input ids과 attention mask는 동일한 구조를 가집니다.
print(
    f"Attention Mask: {inputs['attention_mask']}"
)  # 어텐션 마스크 출력  1: 실제 토큰, 0: 패딩 토큰

# torch.Size([1, 13]) : 배치 크기 1, 시퀀스 길이 13
# 배치 크기 : 한 번에 처리하는 문장 개수
# 시퀀스 길이 : 토큰 개수 (여기서는 max_length로 20 설정)
# 현재 우리가 입력한 한 문장이 토크나이저에 의해 13개의 토큰으로 쪼개졌음을 의미합니다.
# 이는 bert 모델이 입력으로 (batch_size, sequence_length) 형태의 2D 텐서를 받는 것과 일치합니다.
print(f"Input IDs shape(구조): {inputs['input_ids'].shape}")  # 모델 최대 길이 출력

# 4. 숫자를 다시 문자로 복구해보기 (Decoding)
# 토크나이저가 단어를 어떻게 쪼개서 인식했는지 눈으로 확인하는 가장 좋은 방법입니다.
# convert_ids_to_tokens : 숫자 ID를 토큰 단위로 변환
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("\n[2. 토큰 단위 분해]")
print(tokens)

# 실제 문장으로 복구
# decode : 숫자 ID 시퀀스를 다시 문장으로 변환
# unk : unknown 토큰 (사전에 없는 단어)
decoded_sentence = tokenizer.decode(inputs["input_ids"][0])
print("\n[3. 디코딩 결과(특수 토큰 포함)]")
print(decoded_sentence)
