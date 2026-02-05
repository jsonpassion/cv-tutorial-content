# CLIP

> 대조 학습으로 이미지-텍스트 연결

## 개요

이 섹션에서는 멀티모달 AI의 판도를 바꾼 **CLIP(Contrastive Language-Image Pre-training)**을 깊이 있게 살펴봅니다. OpenAI가 2021년에 공개한 이 모델은 4억 개의 이미지-텍스트 쌍으로 학습되어, 한 번도 본 적 없는 카테고리도 분류할 수 있는 놀라운 **zero-shot** 능력을 보여주었습니다.

**선수 지식**: [멀티모달 학습 개론](./01-multimodal-learning.md)의 대조 학습 개념, [Vision Transformer](09-vision-transformer/03-vit.md)의 기본 구조
**학습 목표**:
- CLIP의 듀얼 인코더 아키텍처를 이해한다
- 대조 학습의 학습 목표(loss function)를 설명할 수 있다
- zero-shot 분류가 어떻게 작동하는지 코드로 구현할 수 있다
- CLIP의 후속 모델들(SigLIP, OpenCLIP)을 비교할 수 있다

## 왜 알아야 할까?

여러분이 AI에게 "고양이 사진을 찾아줘"라고 하면 어떻게 될까요? 기존 모델은 "고양이"라는 라벨로 학습된 데이터가 있어야만 찾을 수 있었습니다. 그런데 "아이스크림을 먹고 있는 코알라"를 찾으라고 하면요? 이런 조합으로 학습된 데이터는 아마 없을 겁니다.

CLIP은 이 한계를 극복한 모델입니다. 이미지와 텍스트를 같은 공간에 배치하는 법을 배웠기 때문에, **어떤 텍스트 설명이든** 가장 잘 맞는 이미지를 찾아낼 수 있죠. 이 능력 덕분에 CLIP은 이후 등장한 거의 모든 Vision-Language 모델의 **비전 인코더**로 채택되었습니다. Stable Diffusion, DALL-E, LLaVA 같은 생성/이해 모델뿐 아니라, [Grounding DINO](../07-object-detection/05-detr.md)는 CLIP의 텍스트-이미지 정렬 능력을 활용해 **텍스트로 객체를 탐지**하고, [Grounded SAM](../08-segmentation/04-sam.md)은 여기에 SAM을 결합해 **텍스트로 세그멘테이션**까지 수행합니다. CLIP이 비전 AI 전반의 "눈" 역할을 하고 있는 셈이죠.

## 핵심 개념

### 개념 1: 듀얼 인코더 아키텍처

> 💡 **비유**: CLIP은 마치 **두 명의 통역사**가 일하는 시스템입니다. 한 명은 그림을 보고 특징을 256차원 숫자로 정리하고, 다른 한 명은 글을 읽고 역시 256차원 숫자로 정리합니다. 두 통역사가 같은 언어(같은 벡터 공간)를 사용하기 때문에, 그림과 글의 의미를 직접 비교할 수 있는 거죠.

CLIP의 구조는 놀라울 정도로 단순합니다:

| 구성 요소 | 역할 | 상세 |
|----------|------|------|
| **이미지 인코더** | 이미지 → 임베딩 벡터 | ViT-L/14 (3억 파라미터) 또는 ResNet 변형 |
| **텍스트 인코더** | 텍스트 → 임베딩 벡터 | Transformer (6300만 파라미터), 12층, BPE 토크나이저 |
| **투영 레이어** | 두 임베딩을 같은 차원으로 | 선형 변환으로 공유 공간(512차원)에 매핑 |

이 두 인코더는 서로 완전히 독립적으로 작동합니다. 이미지 인코더는 이미지만 처리하고, 텍스트 인코더는 텍스트만 처리하죠. 두 결과가 만나는 지점은 오직 **유사도 계산** 시점뿐입니다. 이런 구조를 **듀얼 인코더(Dual Encoder)** 아키텍처라고 합니다.

### 개념 2: 대조 학습의 원리

CLIP의 학습 방법은 직관적입니다. 배치에 $N$개의 이미지-텍스트 쌍이 있다면:

- **맞는 쌍** ($N$개): 원래 함께 있던 이미지-텍스트 → 유사도를 **높이기**
- **안 맞는 쌍** ($N^2 - N$개): 잘못 조합된 쌍 → 유사도를 **낮추기**

수식으로 표현하면, 이미지 $i$에 대한 손실 함수는:

$$L_i = -\log \frac{\exp(\text{sim}(I_i, T_i) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(I_i, T_j) / \tau)}$$

- $\text{sim}(I_i, T_j)$: 이미지 $i$와 텍스트 $j$의 코사인 유사도
- $\tau$: 온도 파라미터 (학습 가능, 유사도 분포의 날카로움 조절)
- 분자: 맞는 쌍의 유사도
- 분모: 모든 텍스트와의 유사도 합

이것은 사실 **소프트맥스 크로스 엔트로피**와 같은 형태입니다. "N개 텍스트 중 이 이미지에 맞는 텍스트를 고르시오"라는 N-way 분류 문제를 푸는 것이죠. 텍스트 쪽에서도 대칭적으로 같은 손실을 계산하여, 최종 손실은 두 방향의 평균입니다.

> 💡 **비유**: 배치 크기가 32,768이라면, 매 스텝마다 "이 이미지의 짝은 32,768개 문장 중 어느 것?" 이라는 거대한 객관식 시험을 치르는 셈입니다. 정답률을 높이려면 이미지와 텍스트의 의미를 정말 깊이 이해해야 하겠죠.

### 개념 3: Zero-Shot 분류의 마법

CLIP의 가장 혁명적인 능력은 **zero-shot 분류**입니다. 학습할 때 "고양이", "강아지" 같은 라벨을 전혀 보지 않았는데도, 텍스트 프롬프트만으로 분류가 가능합니다.

작동 원리는 이렇습니다:

1. 분류하고 싶은 카테고리를 텍스트로 변환: "a photo of a cat", "a photo of a dog", ...
2. 각 텍스트의 임베딩을 계산
3. 입력 이미지의 임베딩을 계산
4. 이미지 임베딩과 각 텍스트 임베딩의 코사인 유사도를 계산
5. 가장 유사도가 높은 텍스트의 카테고리가 예측 결과

> ⚠️ **흔한 오해**: "zero-shot이니까 아무 학습 없이 되는 거다" — 아닙니다! CLIP은 4억 개의 이미지-텍스트 쌍으로 **엄청나게 많이** 학습했습니다. "zero-shot"이라는 건 특정 분류 태스크에 대해 추가 학습 없이 작동한다는 의미이지, 전혀 학습하지 않았다는 뜻이 아닙니다.

텍스트 프롬프트의 형식도 중요합니다. 단순히 "cat"보다 "a photo of a cat"이 훨씬 잘 작동하는데, 이는 CLIP이 학습할 때 본 인터넷 캡션이 대부분 문장 형태였기 때문입니다.

### 개념 4: CLIP의 후속 모델들

CLIP 이후 수많은 개선 모델이 등장했습니다:

| 모델 | 개발사 | 핵심 개선 |
|------|--------|----------|
| **OpenCLIP** | LAION 커뮤니티 | 오픈소스 재구현, LAION-2B로 학습, 재현 가능한 스케일링 법칙 연구 |
| **SigLIP** | Google | Sigmoid 손실 함수로 소프트맥스 대체 → 배치 크기에 덜 의존적 |
| **SigLIP 2** | Google (2025) | 셀프 디스틸레이션, 마스크 예측 추가 → 다국어 + 더 정밀한 지역 이해 |
| **EVA-CLIP** | BAAI | EVA 사전학습으로 더 강력한 비전 인코더 |
| **MetaCLIP** | Meta | 데이터 큐레이션 개선, 잡음 제거 |

특히 **SigLIP**은 손실 함수를 소프트맥스에서 시그모이드로 바꾼 것인데요, 이게 왜 중요하냐면 소프트맥스 방식은 배치 내 모든 샘플과 비교해야 하므로 배치 크기가 성능에 큰 영향을 미칩니다. 시그모이드 방식은 각 쌍을 독립적으로 판단하므로 이 제약이 줄어들죠. 2025년 현재 많은 최신 VLM이 CLIP 대신 SigLIP을 비전 인코더로 선택하고 있습니다.

## 실습: 직접 해보기

### 기본: CLIP으로 zero-shot 이미지 분류

```python
import torch
import clip
from PIL import Image

# CLIP 모델 로드 (최초 실행 시 자동 다운로드)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 이미지 로드 및 전처리
image = preprocess(Image.open("test_image.jpg")).unsqueeze(0).to(device)

# 분류할 카테고리를 자연어로 정의
categories = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text = clip.tokenize(categories).to(device)

# 이미지와 텍스트 임베딩 추출
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 정규화 후 유사도 계산
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 결과 출력
for i, category in enumerate(categories):
    print(f"{category}: {similarity[0][i].item():.2%}")
# 예시 출력: a photo of a cat: 92.15%, a photo of a dog: 5.32%, a photo of a bird: 2.53%
```

### 심화: 이미지-텍스트 유사도 행렬 시각화

```python
import torch
import clip
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 여러 텍스트 설명 준비
texts = [
    "a diagram of a neural network",
    "a photo of a sunset over the ocean",
    "a screenshot of a code editor",
    "a painting of a cat",
]
text_tokens = clip.tokenize(texts).to(device)

# 텍스트 임베딩 추출 및 정규화
with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

# 텍스트 간 유사도 행렬 계산
similarity = (text_features @ text_features.T).cpu().numpy()

print("텍스트 간 유사도 행렬:")
print(np.round(similarity, 3))
# 대각선은 1.0 (자기 자신), 의미가 비슷한 쌍일수록 값이 높음
# "neural network"와 "code editor"는 기술 관련이라 유사도가 높고,
# "sunset"과 "cat painting"은 다른 주제라 유사도가 낮을 것
```

### 실전: Hugging Face Transformers로 CLIP 사용하기

```python
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

# Hugging Face에서 CLIP 모델 로드
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 이미지와 텍스트 준비
image = Image.open("example.jpg")
texts = ["a cat sitting on a couch", "a dog playing in the park", "a car on the road"]

# 전처리 및 추론
inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

with torch.no_grad():
    outputs = model(**inputs)

# 이미지-텍스트 유사도 (logits_per_image)
logits = outputs.logits_per_image  # [1, 3]
probs = logits.softmax(dim=1)

for text, prob in zip(texts, probs[0]):
    print(f"{text}: {prob.item():.2%}")
```

## 더 깊이 알아보기

### CLIP 탄생 이야기

CLIP 논문의 정식 제목은 "Learning Transferable Visual Models From Natural Language Supervision"(자연어 감독으로 전이 가능한 비전 모델 학습하기)입니다. 핵심 아이디어는 의외로 간단했습니다. 인터넷에는 이미 수억 개의 이미지-텍스트 쌍(SNS 게시물, 상품 설명, 뉴스 기사의 사진과 캡션 등)이 존재하는데, 이걸 그대로 학습에 활용하자는 것이죠.

기존에는 ImageNet처럼 사람이 직접 라벨을 붙인 고품질 데이터가 필요했지만, CLIP은 인터넷의 "자연어 감독(natural language supervision)"만으로 학습합니다. 이 접근법 덕분에 4억 개라는 방대한 데이터를 사용할 수 있었고, 결과적으로 놀라운 범용성을 얻게 된 것입니다.

OpenAI의 Alec Radford, Jong Wook Kim 등이 주도한 이 연구는 2021년 ICML에서 발표되었고, 발표 당시 ImageNet에서의 zero-shot 정확도가 기존 최고 성능의 지도학습 모델(ResNet-50)과 대등한 수준이라는 점에서 큰 충격을 주었습니다.

### WIT 데이터셋의 비밀

CLIP을 학습시킨 **WebImageText(WIT)** 데이터셋은 공개되지 않았습니다. OpenAI는 50만 개의 검색 쿼리를 사용해 인터넷에서 이미지-텍스트 쌍을 수집했다고만 밝혔는데요, 이 데이터의 비공개가 이후 **OpenCLIP** 프로젝트가 탄생한 계기가 되었습니다. LAION 커뮤니티는 오픈소스 데이터셋(LAION-400M, LAION-2B)을 구축하고 CLIP을 재현하여, 누구나 사용할 수 있는 대안을 만들어냈습니다.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "CLIP은 어떤 이미지든 완벽하게 이해한다" — CLIP은 인터넷 데이터로 학습되었기 때문에, 의료 영상이나 위성 사진 같은 전문 영역에서는 성능이 크게 떨어질 수 있습니다. 또한 텍스트의 위치 관계("왼쪽의 고양이와 오른쪽의 강아지")를 제대로 구분하지 못하는 한계가 있습니다.

> 🔥 **실무 팁**: zero-shot 분류 시 프롬프트 엔지니어링이 성능에 큰 영향을 줍니다. "cat" 대신 "a photo of a cat", 더 나아가 "a bright photo of a cat, high quality"처럼 구체적으로 쓰면 성능이 향상됩니다. OpenAI는 80개의 프롬프트 템플릿을 앙상블하는 방법도 제안했습니다.

> 💡 **알고 계셨나요?**: CLIP의 가장 큰 모델(ViT-L/14)을 학습하는 데 256개의 V100 GPU로 12일이 걸렸습니다. 이를 단일 GPU로 환산하면 약 8.4년이 소요되는 셈이죠. 이런 규모의 학습이 가능했기에 CLIP의 놀라운 범용성이 만들어진 것입니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 듀얼 인코더 | 이미지 인코더와 텍스트 인코더가 독립적으로 작동, 유사도로 연결 |
| 대조 손실 | 맞는 쌍의 유사도 최대화, 안 맞는 쌍의 유사도 최소화 (InfoNCE 손실) |
| Zero-shot 분류 | 카테고리를 텍스트 프롬프트로 변환하여 유사도 기반 분류 |
| 온도 파라미터 $\tau$ | 유사도 분포의 날카로움 조절, 학습 가능 파라미터 |
| SigLIP | Sigmoid 손실로 배치 크기 의존성 해결, 2025년 주류 |
| OpenCLIP | 오픈소스 CLIP 재구현, LAION 데이터로 학습 |

## 다음 섹션 미리보기

CLIP은 "이미지와 텍스트를 매칭"하는 데는 뛰어나지만, 이미지를 보고 **자연스러운 문장을 생성**하는 것은 별개의 문제입니다. 다음 섹션 [BLIP과 BLIP-2](./03-blip.md)에서는 이 한계를 넘어 이미지 캡셔닝, 시각적 질의응답 등 **생성 능력**까지 갖춘 모델을 만나봅니다.

## 참고 자료

- [Learning Transferable Visual Models From Natural Language Supervision (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) - CLIP 원본 논문, 모든 VLM의 출발점
- [CLIP: Connecting text and images (OpenAI Blog)](https://openai.com/index/clip/) - OpenAI의 CLIP 공식 소개 페이지
- [OpenCLIP GitHub Repository](https://github.com/mlfoundations/open_clip) - 오픈소스 CLIP 구현과 다양한 사전학습 모델
- [SigLIP 2: Multilingual Vision-Language Encoders (Google, 2025)](https://arxiv.org/abs/2502.14786) - Sigmoid 기반 CLIP의 최신 진화
- [Zero-shot Image Classification with CLIP (Pinecone)](https://www.pinecone.io/learn/series/image-search/zero-shot-image-classification-clip/) - CLIP zero-shot 분류의 실용적 튜토리얼
