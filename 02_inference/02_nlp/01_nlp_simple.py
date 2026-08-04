"""
pipline을 사용한 간단한 NLP 태스크로 감성 분석, 개체명 인식, 요약에서 동일한 토크나이저를 사용하는 예제입니다.
"""

import torch
from transformers import pipeline

"""
uv add  hf_xet
"""

# GPU 사용 여부 체크
device = 0 if torch.cuda.is_available() else -1

# 1. 감성 분석 (Sentiment Analysis)
# 모델: distilbert-base-uncased-finetuned-sst-2-english (가볍고 빠르며 정확도가 높은 표준 모델)
# 링크: https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english
sentiment_app = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device,
)

# 2. 개체명 인식 (NER): 인물(PER), 조직(ORG), 장소(LOC) 등 추출
# 모델: dslim/bert-base-NER (CoNLL-2003 데이터셋으로 학습된 가장 대중적인 NER 모델)
# 링크: https://huggingface.co/dslim/bert-base-NER
ner_app = pipeline(
    "token-classification",
    model="dslim/bert-base-NER",
    device=device,
    aggregation_strategy="simple",
)

# 3. 요약 (Summarization)
# 모델: facebook/bart-large-cnn (뉴스 요약에 최적화된 고성능 BART 모델)
# 링크: https://huggingface.co/facebook/bart-large-cnn
summarizer = pipeline("text-generation", model="facebook/bart-large-cnn", device=device)

print("--- [All English Models Loaded Successfully] ---")

# 테스트 데이터
review = "I bought a MacBook Pro at the Gangnam store yesterday, but there is a black dot on the screen. I want a full refund. The staff was very unkind."
# "어제 강남 매장에서 맥북 프로를 샀는데 화면에 검은 점이 있습니다. 전액 환불을 원합니다. 직원들이 매우 불친절했습니다."

# 실행
print("--- [Analysis Results] ---")
print(f"1. Sentiment: {sentiment_app(review)}")
print(f"2. NER: {ner_app(review)}")
# 요약의 경우 입력 문장이 짧으면 원문이 그대로 나올 수 있으므로, 파라미터를 조절합니다.
# 아래의 모델은 정확하지 않을 수 있음.
print(f"3. Summarization: {summarizer(review, max_length=30, min_length=5)}")
