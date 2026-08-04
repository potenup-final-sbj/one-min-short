from transformers import pipeline

"""
Question Answering
질문 답변 태스크에서는 모델이 주어진 문서(컨텍스트)를 바탕으로 사용자의 질문에 대한 답변을 찾아야 합니다.
예를 들어, "제미니는 누구에 의해 개발되었나요?"라는 질문이 주어지고,
"제미니는 구글에서 개발한 고성능 AI 모델입니다."라는 문서가 주어지면,
모델은 "구글"이라는 답변을 반환해야 합니다.

bert는 구조적으로 인코더 전용 모델로 입력 받은 텍스트의 의미를 잘 파악하는 데 강점이 있습니다.
따라서 질문 답변 태스크에서 뛰어난 성능을 발휘하지만 생성에는 한계가 있습니다.
"""

# 이번에는 구체적인 모델을 지정하여 파이프라인을 만듭니다.
# 한국어 질문 답변 모델을 사용해 보겠습니다.
# "deepset/roberta-base-squad2" 모델은 RoBERTa 기반의 사전학습 모델로, SQuAD2.0 데이터셋으로 미세조정된 모델입니다.
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

print("--- [Pipeline 내부 구조 분석] ---")
# 1. 내부 모델 확인
# qa_pipeline.model : 파이프라인 내부에 로드된 모델 객체
print(f"1. 내부 모델 클래스: {type(qa_pipeline.model)}")

# 2. 내부 토크나이저 확인
# qa_pipeline.tokenizer : 파이프라인 내부에 로드된 토크나이저 객체
print(f"2. 내부 토크나이저 클래스: {type(qa_pipeline.tokenizer)}")

# 3. 실제 질문 던져보기
context = "Gemini is a highly capable AI model developed by Google. It excels in reasoning and coding tasks."
# 제미니는 구글에서 개발한 고성능 AI 모델입니다. 추론 및 코딩 작업에서 탁월합니다
question = "Who developed Gemini?"
# 제미니는 누가 개발했나요?

# question : 질문 텍스트
# context : 질문의 근거가 되는 문서(컨텍스트)
result = qa_pipeline(question=question, context=context)

print(f"\n3. 질문: {question}")
print(f"   답변: {result['answer']}")
print(f"   위치: {result['start']} ~ {result['end']} (텍스트 내 인덱스)")
