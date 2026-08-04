import requests
from PIL import Image
from transformers import pipeline

"""
uv add Pillow timm

Pillow 
이미지 처리를 위한 파이썬 라이브러리로, 다양한 이미지 파일 형식을 지원하고
이미지 열기, 저장, 변환 등의 기능을 제공합니다.

timm
PyTorch 기반의 이미지 모델 라이브러리로, 다양한 사전학습된 비전 모델을 제공하여
이미지 분류, 객체 탐지, 세분화 등의 작업에 활용할 수 있습니다
"""

# 1. 객체 탐지 파이프라인 로드 (DEtection TRansformer - DETR 모델 활용)
# 'detr-resnet-50'은 성능과 속도의 균형이 좋은 실무용 모델입니다.
detector = pipeline("object-detection", model="facebook/detr-resnet-50")

# 2. 테스트 이미지 준비 (샘플 이미지 URL)
url = "http://images.cocodataset.org/val2017/000000039769.jpg"

# requests.get(url, stream=True): 인터넷에 있는 이미지를 가져오는데
# stream=True 옵션은 응답 내용을 스트리밍 방식으로 처리하여 메모리 사용을 최적화합니다.
# .raw 속성은 응답 객체에서 원시 바이트 스트림에 접근할 수 있게 합니다.
# Image.open(): PIL 라이브러리를 사용하여 이미지를 엽니다.
# 즉, 인터넷에서 이미지를 가져오는데 실시간으로 바이트 스트림을 읽어와서
# PIL 이미지 객체로 변환하는 과정입니다.
image = Image.open(requests.get(url, stream=True).raw)

# 3. 추론 수행
# threshold=0.9: 90% 이상의 확신이 있는 객체만 뽑습니다.
results = detector(image, threshold=0.9)
print(f"--- [총 {len(results)}개의 객체가 탐지되었습니다] ---")

# 4. 탐지된 객체 크롭 및 출력
for i, res in enumerate(results):
    box = res["box"]  # 객체의 경계 상자 좌표
    label = res["label"]  # 객체 클래스 라벨
    score = res["score"]  # 객체 탐지 확신도

    # PIL의 crop 함수 사용: (left, upper, right, lower) 튜플 형태 전달
    # 좌표값이 딕셔너리 키와 일치하므로 바로 추출하여 전달합니다.
    crop_box = (box["xmin"], box["ymin"], box["xmax"], box["ymax"])

    # image.crop(): 이미지에서 지정된 영역을 잘라냅니다.
    # x, y 좌표를 사용하여 객체가 포함된 부분만 크롭합니다.
    cropped_img = image.crop(crop_box)

    # 결과 출력
    print(f"[{i+1}] {label} (신뢰도: {round(score, 4)})")

    # 크롭된 이미지 보기 (로컬 환경에서 새 창으로 뜸)
    cropped_img.show()

    # 만약 파일로 저장하고 싶다면 아래 주석 해제
    # cropped_img.save(f"detected_{label}_{i}.jpg")

print("--- [크롭 작업 완료] ---")
