"""
01_nlp-simple.py에서는 파이프라인 API를 사용하여
감성 분석, 개체명 인식, 요약 태스크를 각각 독립적으로 수행하는 예제를 다루었습니다.
이번 02_nlp.py에서는 동일한 '베이스 모델'과 '토크나이저'를 공유하면서
각 태스크별로 다른 '헤드(Head)'만 교체하여 사용하는 방법을 살펴보겠습니다.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    EncoderDecoderModel,
    pipeline,
)

"""
uv add  hf_xet
"""


# 1. 공통 토크나이저 (The Same Brain Language)
# 모든 태스크는 동일한 'bert-base-uncased' 토크나이저를 공유합니다.
model_name = "bert-base-uncased"
shared_tokenizer = AutoTokenizer.from_pretrained(model_name)


# GPU 설정
device = 0 if torch.cuda.is_available() else -1
device_str = "cuda" if torch.cuda.is_available() else "cpu"

print(f"--- [공통 토크나이저 로드 완료: {model_name}] ---")


# A. 감성 분석: BERT Base + Sequence Classification Head
# https://huggingface.co/textattack/bert-base-uncased-SST-2

"""
AutoModelForSequenceClassification
squence classification(문장 분류) 태스크에 맞게 사전학습된 BERT 모델을 로드하기 위한 클래스입니다.
이는 BERT Base 모델 위에 분류용 헤드(Linear Layer)가 추가된 구조를 가집니다.
"""
sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/bert-base-uncased-SST-2"
).to(device_str)


# B. 개체명 인식(NER): BERT Base + Token Classification Head
# https://huggingface.co/dslim/bert-base-NER
"""
AutoModelForTokenClassification
token classification(토큰 분류) 태스크에 맞게 사전학습된 BERT 모델을 로드하기 위한 클래스입니다.
이는 BERT Base 모델 위에 토큰별 분류용 헤드(Linear Layer)가 추가된 구조를 가집니다.
"""
ner_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER").to(
    device_str
)

# C. 요약(Summarization): BERT Base를 Encoder와 Decoder로 모두 사용한 구조
# BERT의 아키텍처를 그대로 유지하면서 요약이 가능하게 만든 모델입니다.
# https://huggingface.co/patrickvonplaten/bert2bert-cnn_dailymail-fp16
"""
EncoderDecoderModel
Encoder-Decoder(Seq2Seq) 구조를 구현한 모델 클래스로, 입력 문장을 이해하는 인코더와
출력 문장을 생성하는 디코더로 구성됩니다.
이는 BERT Base 모델을 인코더와 디코더로 모두 활용할 수 있습니다.
"""
summarization_model = EncoderDecoderModel.from_pretrained(
    "patrickvonplaten/bert2bert-cnn_dailymail-fp16"
).to(device_str)

"""
모든 모델이 Bert Base를 기반으로 하고 있다는 것을 알 수 있다.
이는 모델의 기본 레이어(베이스 모델)가 동일하다는 의미하며
outputs(헤드)만 다르다는 것을 뜻한다.
"""

# 3. Pipeline 구축 (공통된 tokenizer를 명시적으로 주입)
# 여기서 중요한 점은 모든 pipeline에 'tokenizer=shared_tokenizer'가 들어간다는 것입니다.
# 같은 토크나이저를 사용함으로써 입력 텍스트가 동일한 방식으로 토큰화되어 모델에 전달됩니다.

# 감성 분석 파이프라인
sentiment_app = pipeline(
    "sentiment-analysis",  # 태스크 이름
    model=sentiment_model,  # 모델 객체
    tokenizer=shared_tokenizer,  # 공통 토크나이저
    device=device,  # 실행 장치 지정
)

# 개체명 인식 파이프라인
ner_app = pipeline(
    "token-classification",  # 태스크 이름
    model=ner_model,  # 모델 객체
    tokenizer=shared_tokenizer,  # 공통 토크나이저
    device=device,  # 실행 장치 지정
    aggregation_strategy="simple",  # 개체명 인식 결과 병합 전략
    # simple: 인접한 동일 개체명 병합, first : 첫 토큰 기준, max: 확률 최대 토큰 기준, average: 확률 평균 기준
)

# 요약 파이프라인 (BERT2BERT 모델 사용)
summarizer = pipeline(
    "summarization",  # 태스크 이름
    model=summarization_model,  # 모델 객체
    tokenizer=shared_tokenizer,  # 공통 토크나이저
    device=device,  # 실행 장치 지정
)

print("--- [모든 BERT 기반 Task별 Head 로드 완료] ---\n")

# 테스트 데이터
review = "I bought a MacBook Pro at the Gangnam store yesterday, but there is a black dot on the screen. I want a full refund. The staff was very unkind."

# 실행
print("--- [분석 결과] ---")
print("--- [Sentiment Analysis Head] ---")
print(f"1. 감성 분석 (Head: Classifier): \n   {sentiment_app(review)}\n")
print(sentiment_model.classifier)
# 출력 예시: Linear(in_features=768, out_features=2, bias=True)
# out_features=2 : 긍정/부정 2가지 클래스로 분류

print("\n--- [NER Head] ---")
print(f"2. 개체명 인식 (Head: Token Classifier): \n   {ner_app(review)}\n")
print(ner_model.classifier)
# 출력 예시: Linear(in_features=768, out_features=9, bias=True)
# out_features=9 : 9가지 개체명 클래스로 분류
# 출력 태그: 'PER'(인물), 'ORG'(조직), 'LOC'(장소), 'MISC'(기타)


print("\n--- [Summarization Head] ---")
print(
    f"3. 요약 (Head: Seq2Seq LM): \n   {summarizer(review, max_length=25, min_length=5)}"
)
print(
    summarization_model.decoder
)  # 요약 모델은 output layer가 LM 헤드로 seq2seq 구조를 갖게 된다.
# 출력 예시: BertLMHeadModel(config=BertConfig {
# 요약 생성용 LM 헤드


"""
base model은 모든 태스크에서 동일하게 bert-base-uncased를 사용하고 있으며,
동일한 base model을 사용한다는 것은 hidden layer의 구조와 가중치가 동일하다는 것을 의미한다.

또한 동일한 tokenizer를 공유하고 있다는 것은 모델의 input layer도 동일하다는 것을 의미한다.

따라서 각 태스크별로 다른 부분은 output layer, 즉 head 부분만 태스크별로 다르게 구성되어 있음을 알 수 있다.
"""
