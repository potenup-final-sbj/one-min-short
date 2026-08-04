import torch
from transformers import pipeline


class VOCAnalyzer:

    def __init__(self):
        print("AI 모델들을 로드 중입니다... (VRAM 점유 시작)")
        self.device = 0 if torch.cuda.is_available() else -1

        # 1. 감성 분석: distilbert-base-uncased-finetuned-sst-2-english
        # 출력 라벨: 'POSITIVE', 'NEGATIVE'
        self.sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=self.device,
        )

        # 2. 개체명 인식: dslim/bert-base-NER
        # 출력 태그: 'PER'(인물), 'ORG'(조직), 'LOC'(장소), 'MISC'(기타)
        self.ner = pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            device=self.device,
            aggregation_strategy="max",
        )

        # 3. 요약: facebook/bart-large-cnn
        # 뉴스 데이터에 최적화된 고성능 요약 모델
        self.summarizer = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=self.device
        )

    def analyze(self, text):
        # 1. 감성 분석 수행
        sent_res = self.sentiment(text)[0]

        # 2. 핵심 키워드(개체명) 추출
        # 영어 NER 모델의 태그인 'ORG', 'LOC', 'PER'를 기준으로 필터링합니다.
        ner_res = self.ner(text)
        # ner_res에서 추출한 값 중 'entity_group'가 'ORG', 'LOC', 'PER'인 개체의 'word' 값을 리스트로 만듭니다.
        keywords = [
            entity["word"]
            for entity in ner_res
            if entity["entity_group"] in ["ORG", "LOC", "PER"]
        ]

        # 3. 요약 수행 (입력 텍스트가 50자 이상일 때만 수행)
        if len(text) > 50:
            # 영어 문장은 한국어보다 길어지는 경향이 있어 max_length를 40으로 조정했습니다.
            summary = self.summarizer(text, max_length=40, min_length=10)[0][
                "summary_text"
            ]
        else:
            summary = text

        return {
            # DistilBERT 모델은 부정적일 때 'NEGATIVE' 라벨을 반환합니다.
            "is_angry": sent_res["label"] == "NEGATIVE",
            "score": round(sent_res["score"], 2),  # 감정 분석 확신도
            "mentions": list(set(keywords)),  # 추출된 개체명(중복 제거)
            "tldr": summary,  # 요약문
        }


# --- 실무 시나리오 적용 ---
voc_engine = VOCAnalyzer()

# 테스트 데이터 (영어 버전)
sample_voc = """
I bought a Samsung Galaxy S24 at the Lotte Mall store in Jamsil last Saturday. 
However, the battery drains too quickly and it overheats so much that I'm afraid to use it. 
I called the customer center, but I couldn't get through and I'm very angry. Please take action.
"""

report = voc_engine.analyze(sample_voc)

print("\n" + "=" * 30)
print("     VOC ANALYSIS REPORT     ")
print("=" * 30)
print(f"Sentiment Status: {'🚨 ANGRY / DANGER' if report['is_angry'] else '✅ NORMAL'}")
print(f"Confidence Score: {report['score']}")
print(f"Entities Found  : {report['mentions']}")
print(f"Summary (TL;DR) : {report['tldr']}")
print("=" * 30)
