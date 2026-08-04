import torch
from transformers import pipeline

# 1. 실행 장치 확인
# torch.cuda.is_available() : 디바이스 내에 GPU가 있는지 확인하는 함수
# device=0은 첫 번째 GPU를 의미하며, 1은 두 번째 GPU, GPU가 없으면 -1(CPU)을 사용합니다.
device = 0 if torch.cuda.is_available() else -1
print(f"사용 중인 장치: {'GPU' if device == 0 else 'CPU'}")

# 2. Pipeline 인스턴스 생성 (Facade Pattern의 정수)
# "sentiment-analysis" : 감정 분석(긍정/부정) 태스크용 파이프라인
# 허깅페이스가 가장 적절한 기본 모델(distilbert-base-uncased-finetuned-sst-2-english)을 로드합니다.
# device 매개변수를 통해 실행 장치를 지정합니다.
classifier = pipeline("sentiment-analysis", device=device)

# 3. 데이터 입력 및 결과 출력
# 리스트 형태로 여러 문장을 한꺼번에 전달할 수 있습니다.
results = classifier(
    [
        "This course is absolutely life-changing for junior developers!",  # 긍정 문장
        "I'm a bit disappointed with the lack of advanced examples.",  # 부정 문장
    ]
)

print("\n--- [추론 결과] ---")
# zip : 여러 개의 이터러블(반복 가능한 객체)을 병렬로 순회할 때 사용
for text, res in zip(
    [
        "This course is absolutely life-changing for junior developers!",
        "I'm a bit disappointed with the lack of advanced examples.",
    ],
    results,
):
    print(f"문장: {text}")
    print(f"결과: {res['label']} (확률: {res['score']:.4f})")
    print("-" * 20)
