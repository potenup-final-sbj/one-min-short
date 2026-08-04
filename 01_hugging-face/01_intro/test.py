# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-classification", model="google/mobilenet_v2_1.4_224")

results = pipe(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/hub/parrots.png"
)

print(results)

"""
[{'label': 'macaw', 'score': 0.7889607548713684}, 
{'label': 'quail', 'score': 0.07045435905456543}, 
{'label': 'partridge', 'score': 0.021508391946554184}, 
{'label': 'vulture', 'score': 0.018251385539770126}, 
{'label': 'hornbill', 'score': 0.017857331782579422}]
"""
