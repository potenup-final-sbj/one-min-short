# 이미지 분류 파이프라인 예제
# mobilenet은 경량화된 모델로, 모바일 및 임베디드 장치에서 효율적으로 동작하도록 설계되었습니다.
# 주로 이미지 분류 작업에 사용되며, 적은 계산 자원으로도 높은 정확도를 제공합니다.
# 총 1000개의 클래스(예: 개, 고양이, 자동차 등)로 이미지를 분류할 수 있습니다.

# 시작하기 전에
# uv add pillow
# pillow : Python Imaging Library(PIL)의 포크 버전으로, 이미지 처리 기능을 제공합니다.
from transformers import pipeline

"""
pipeline 
Hugging Face Transformers 라이브러리에서 제공하는 고수준 API로, 
복잡한 모델 로딩, 전처리, 후처리 과정을 단순화하여 한 줄의 코드로 
다양한 NLP 및 컴퓨터 비전 작업을 수행할 수 있게 해줍니다.

huggingface의 무기는 바로 이 pipeline입니다.
"""
# pipeline : 첫번째 인자로 작업 유형을 받고, 두번째 인자로 모델 이름을 받습니다.
# 내부적으로 AutoModel, AutoTokenizer 등을 사용하여 모델과 토크나이저를 자동으로 로드합니다.
# tokenizer와 model을 일일이 불러올 필요 없이, pipeline 하나로 모든 작업이 끝납니다.
pipe = pipeline("image-classification", model="google/mobilenet_v2_1.4_224")

# 강아지 이미지 URL
# https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSnDrBPgDyqt1BxXJ0vWqC0ZPPoSPt67B_I6YXIXv8c-r00HT7kTITeTsyRtAr4z0hhOg3s5FCsAb38GcyKQosHVCi-VHwr0DbShdGt1PIr&s=10
# 위 url을 넣어도 됨.
result = pipe("../../00_img/dog.jpg")
print(result)
