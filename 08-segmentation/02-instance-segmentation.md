# 인스턴스 세그멘테이션

> Mask R-CNN, YOLACT

## 개요

앞서 [시맨틱 세그멘테이션](./01-semantic-segmentation.md)에서는 모든 픽셀에 클래스 라벨을 부여하는 방법을 배웠습니다. 하지만 한 가지 아쉬운 점이 있었죠 — 사진에 사람이 3명 있어도 모두 "사람"이라는 같은 색으로 칠해져서, **누가 누구인지** 구분할 수 없었습니다. **인스턴스 세그멘테이션(Instance Segmentation)**은 이 문제를 해결합니다. 각 객체를 **개별적으로 구분**하면서, 동시에 **픽셀 단위의 정밀한 마스크**를 예측합니다.

**선수 지식**: [시맨틱 세그멘테이션](./01-semantic-segmentation.md), [R-CNN 계열](../07-object-detection/02-rcnn-family.md)
**학습 목표**:
- 시맨틱 세그멘테이션과 인스턴스 세그멘테이션의 차이를 명확히 이해한다
- Mask R-CNN의 아키텍처와 핵심 기법(RoIAlign)을 설명할 수 있다
- YOLACT 등 실시간 인스턴스 세그멘테이션 접근법을 이해한다

## 왜 알아야 할까?

현실 세계에서는 같은 종류의 객체가 여러 개 동시에 등장하는 경우가 대부분입니다. 자율주행차는 앞에 있는 자동차 3대를 **각각** 추적해야 하고, 로봇은 테이블 위의 컵 5개를 **하나씩** 집어야 하며, 의료 영상에서는 개별 세포 하나하나를 **따로** 분석해야 합니다.

| 태스크 | 클래스 구분 | 개별 객체 구분 | 출력 |
|--------|-----------|--------------|------|
| **시맨틱 세그멘테이션** | ✅ | ❌ | 클래스 맵 (H×W) |
| **인스턴스 세그멘테이션** | ✅ | ✅ | 객체별 마스크 N개 |
| **객체 탐지** | ✅ | ✅ | 바운딩 박스 N개 |

인스턴스 세그멘테이션은 **객체 탐지의 정밀도**와 **세그멘테이션의 디테일**을 동시에 갖춘, 가장 풍부한 정보를 제공하는 태스크입니다.

## 핵심 개념

### 1. 인스턴스 세그멘테이션이란? — "누가 누구인지" 구분하기

> 💡 **비유**: 단체 사진을 찍었을 때, 시맨틱 세그멘테이션은 "사람 영역"을 하나의 색으로 칠합니다. 마치 그룹 사진에서 모든 사람을 하나로 묶어 "여기가 사람들"이라고 표시하는 거죠. 반면 인스턴스 세그멘테이션은 **각 사람마다 다른 색**으로 칠합니다 — "이 사람은 빨강, 저 사람은 파랑, 그 사람은 초록"처럼요. 개별 인물을 구분해서 정확한 윤곽선까지 그려주는 겁니다.

핵심 차이를 다시 정리하면:

- **시맨틱 세그멘테이션**: "이 픽셀은 **무슨 클래스**?" → 클래스 ID 맵
- **인스턴스 세그멘테이션**: "이 픽셀은 **몇 번째 객체의 무슨 클래스**?" → 객체별 마스크 + 클래스 + 신뢰도

인스턴스 세그멘테이션의 접근 방식은 크게 두 가지로 나뉩니다:

- **Top-Down (탐지 기반)**: 먼저 객체를 탐지 → 각 객체 영역 내에서 마스크 예측 (Mask R-CNN)
- **Bottom-Up (그룹핑 기반)**: 먼저 모든 픽셀을 세그멘테이션 → 같은 객체에 속하는 픽셀을 그룹핑

> ⚠️ **흔한 오해**: "인스턴스 세그멘테이션은 배경도 구분한다"고 생각할 수 있지만, 실제로는 **사물(thing) 클래스만** 다룹니다. 하늘, 도로, 풀밭 같은 **재료(stuff) 클래스**는 "인스턴스"라는 개념이 없으므로 인스턴스 세그멘테이션의 대상이 아닙니다. 사물과 재료를 모두 다루려면 [파놉틱 세그멘테이션](./03-panoptic-segmentation.md)이 필요합니다.

### 2. Mask R-CNN — 탐지에 마스크를 더하다

> 💡 **비유**: [Faster R-CNN](../07-object-detection/02-rcnn-family.md)이 "이 사각형 안에 고양이가 있다"고 알려줬다면, Mask R-CNN은 거기에 **형광펜으로 고양이의 정확한 윤곽을 따라 그려주는** 기능을 추가한 것입니다. 탐지와 분류는 그대로 하면서, 마스크 예측 브랜치 하나만 추가한 거죠.

2017년 Kaiming He(ResNet의 아버지!)가 이끄는 Facebook AI Research(FAIR) 팀이 발표한 **Mask R-CNN**은, 인스턴스 세그멘테이션의 대표 모델입니다. 설계 철학이 놀랍도록 심플합니다.

**핵심 아이디어**: Faster R-CNN + **마스크 예측 브랜치** 1개 추가

Faster R-CNN의 구조를 그대로 가져오되, 각 RoI(관심 영역)에 대해 기존의 클래스 분류 + 박스 회귀에 더해, **28×28 크기의 이진 마스크**를 예측하는 브랜치를 병렬로 추가합니다.

**Mask R-CNN의 파이프라인**:

1. **백본(ResNet + FPN)**: 이미지에서 다중 스케일 특징 맵 추출
2. **RPN(Region Proposal Network)**: 객체 후보 영역(RoI) 제안
3. **RoIAlign**: 후보 영역의 특징을 **정밀하게** 추출 (핵심 기여!)
4. **분류 + 박스 회귀 헤드**: 클래스와 바운딩 박스 예측
5. **마스크 헤드**: 각 RoI에 대해 28×28 이진 마스크 예측 (병렬 수행)

**RoIAlign — Mask R-CNN의 숨겨진 영웅**:

기존 Faster R-CNN의 RoIPool은 특징 맵에서 영역을 추출할 때 **양자화(반올림)** 과정에서 미세한 위치 오차가 발생했습니다. 분류에는 큰 문제가 아니지만, **픽셀 단위** 마스크 예측에는 이 오차가 치명적이죠.

RoIAlign은 양자화 대신 **이중선형 보간(Bilinear Interpolation)**을 사용하여, 정확한 위치의 특징을 추출합니다. 이 작은 변화가 마스크 정확도를 크게 끌어올렸습니다.

| 기법 | 방식 | 문제점 |
|------|------|--------|
| **RoIPool** | 좌표를 정수로 반올림 → 가장 가까운 셀 선택 | 미세한 위치 오차 누적 |
| **RoIAlign** | 이중선형 보간으로 정확한 위치 값 계산 | 오차 없음 (마스크 품질 ↑) |

```python
import torch
import torch.nn as nn
from torchvision.ops import RoIAlign

# RoIAlign 사용 예시
roi_align = RoIAlign(
    output_size=(7, 7),    # 출력 크기 고정
    spatial_scale=1/16,    # 입력 대비 특징 맵의 축소 비율
    sampling_ratio=2        # 각 셀에서 샘플링할 포인트 수
)

# 특징 맵과 RoI 준비
features = torch.randn(1, 256, 32, 32)  # 백본 출력
# 각 RoI: [배치 인덱스, x1, y1, x2, y2] (원본 이미지 좌표)
rois = torch.tensor([[0, 50.5, 30.2, 200.7, 180.9]])  # 소수점 좌표 가능!

# RoIAlign 적용
aligned_features = roi_align(features, rois)
print(f"RoI 특징: {aligned_features.shape}")  # [1, 256, 7, 7]
```

**마스크 예측 헤드**:

마스크 헤드는 각 RoI에 대해 **클래스별로 독립적인** 28×28 이진 마스크를 예측합니다. 즉, C개 클래스에 대해 C개의 마스크를 예측하고, 분류 헤드에서 결정한 클래스에 해당하는 마스크만 사용합니다. 이렇게 **분류와 마스크 예측을 분리(decouple)**한 것이 Mask R-CNN의 중요한 설계 결정이었습니다.

```python
import torch
import torch.nn as nn

class MaskHead(nn.Module):
    """Mask R-CNN의 마스크 예측 헤드 (간소화 버전)"""
    def __init__(self, in_ch=256, num_classes=80):
        super().__init__()
        # 4개의 합성곱 레이어로 특징 정제
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_ch, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
        )
        # 전치 합성곱으로 해상도 2배 업샘플 (14→28)
        self.upsample = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.relu = nn.ReLU()
        # 클래스별 독립적인 마스크 예측 (1×1 conv)
        self.mask_pred = nn.Conv2d(256, num_classes, 1)

    def forward(self, roi_features):
        # roi_features: [N, 256, 14, 14] (RoIAlign 출력)
        x = self.conv_layers(roi_features)  # [N, 256, 14, 14]
        x = self.relu(self.upsample(x))     # [N, 256, 28, 28]
        masks = self.mask_pred(x)            # [N, num_classes, 28, 28]
        return masks  # 클래스별 28×28 이진 마스크

# 테스트
mask_head = MaskHead(in_ch=256, num_classes=80)
roi_feats = torch.randn(5, 256, 14, 14)    # 5개 RoI
masks = mask_head(roi_feats)
print(f"마스크 출력: {masks.shape}")          # [5, 80, 28, 28]
# → 5개 객체 × 80개 클래스 × 28×28 마스크
```

### 3. YOLACT — 실시간 인스턴스 세그멘테이션

> 💡 **비유**: Mask R-CNN이 각 객체에 대해 **맞춤 마스크를 하나씩 제작**하는 방식이라면, YOLACT는 **미리 만들어둔 마스크 조각(프로토타입)들을 조합**하여 최종 마스크를 만드는 방식입니다. 레고 블록을 조합하는 것처럼요!

2019년 Daniel Bolya가 발표한 **YOLACT(You Only Look At CoefficienTs)**는 실시간 인스턴스 세그멘테이션의 선구자입니다. 이름에서 눈치채셨나요? [YOLO](../07-object-detection/03-yolo.md)의 "You Only Look" 정신을 계승하면서, "At CoefficienTs"라는 독자적인 아이디어를 결합한 것이죠. Mask R-CNN이 각 RoI마다 마스크를 개별 예측하는 것과 달리, YOLACT는 YOLO처럼 **속도를 최우선**에 두면서 전혀 다른 접근을 택했습니다.

**YOLACT의 핵심 아이디어 — 마스크를 분해하라!**

1. **프로토타입 생성 브랜치(Protonet)**: 이미지 전체에 대해 $k$개의 **프로토타입 마스크**(기본 마스크 패턴)를 생성
2. **계수 예측 브랜치**: 각 탐지된 객체에 대해 $k$개의 **혼합 계수**를 예측
3. **마스크 조립**: 프로토타입 × 계수 → 최종 인스턴스 마스크 (행렬 곱셈 한 번!)

이 방식의 장점은 **마스크 생성이 단순한 행렬 곱셈**이라는 점입니다. RoI별로 별도의 합성곱을 수행하는 Mask R-CNN과 달리, 모든 객체의 마스크를 한 번에 계산할 수 있어 **훨씬 빠릅니다**.

| 모델 | 속도 (FPS) | COCO AP | 접근 방식 |
|------|-----------|---------|----------|
| **Mask R-CNN** | ~5 | 37.1 | RoI별 개별 마스크 예측 |
| **YOLACT** | ~33 | 29.8 | 프로토타입 + 계수 조합 |
| **YOLACT++** | ~34 | 34.1 | + Deformable Conv + Fast NMS |

> 🔥 **실무 팁**: 정확도가 최우선이라면 Mask R-CNN, **실시간 처리**가 필요하다면 YOLACT 계열을 선택하세요. 최근에는 두 장점을 모두 잡으려는 모델들(SOLOv2, Mask2Former 등)도 많이 나왔습니다.

### 4. 박스 없이도 가능하다 — SOLO와 CondInst

Mask R-CNN과 YOLACT는 모두 **먼저 박스를 찾고** 그 안에서 마스크를 예측하는 방식입니다. 하지만 [Anchor-Free 탐지기](../07-object-detection/04-anchor-free.md)에서 앵커 없이 객체를 찾는 패러다임이 등장한 것처럼, 인스턴스 세그멘테이션에서도 **박스 없이 직접 마스크를 예측**하는 방법이 등장했습니다.

**SOLOv2(2020)**는 [FCOS](../07-object-detection/04-anchor-free.md)처럼 이미지를 격자로 나누고, 각 격자 셀의 객체에 대해 **동적 커널**과 **마스크 특징**을 예측합니다. 이 둘을 합성곱하면 인스턴스 마스크가 됩니다. 바운딩 박스를 전혀 사용하지 않으면서도 Mask R-CNN에 근접한 성능(COCO AP 41.7%)을 달성했습니다.

**CondInst(Conditional Instance Segmentation)**도 비슷한 아이디어로, 각 인스턴스에 대해 **조건부 합성곱 필터**를 생성하고, 이 필터로 공유 특징 맵을 합성곱하여 마스크를 만듭니다.

이런 **Box-Free** 접근법의 장점은 파이프라인이 단순해지고, [NMS](../07-object-detection/01-detection-basics.md) 같은 후처리를 간소화할 수 있다는 점입니다. 객체 탐지에서의 Anchor-Free 혁명이 세그멘테이션으로까지 확장된 셈이죠.

## 실습: torchvision Mask R-CNN 사용하기

torchvision의 사전학습된 Mask R-CNN으로 인스턴스 세그멘테이션을 바로 수행해봅시다.

```python
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights

# 사전학습된 Mask R-CNN v2 로드 (COCO 데이터셋)
weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
model = maskrcnn_resnet50_fpn_v2(weights=weights)
model.eval()

# 임의 이미지로 추론
images = [torch.randn(3, 480, 640)]  # 리스트로 전달 (배치 아님!)
with torch.no_grad():
    predictions = model(images)

# 결과 구조 확인
pred = predictions[0]
print(f"탐지된 객체 수: {len(pred['scores'])}")
print(f"박스: {pred['boxes'].shape}")       # [N, 4] — 바운딩 박스
print(f"라벨: {pred['labels'].shape}")      # [N] — 클래스 ID
print(f"점수: {pred['scores'].shape}")      # [N] — 신뢰도
print(f"마스크: {pred['masks'].shape}")     # [N, 1, H, W] — 인스턴스 마스크

# 신뢰도 높은 객체만 필터링
threshold = 0.5
high_conf = pred['scores'] > threshold
filtered_masks = pred['masks'][high_conf]
filtered_labels = pred['labels'][high_conf]
print(f"\n신뢰도 {threshold} 이상 객체: {high_conf.sum().item()}개")

# COCO 클래스 이름 (일부)
COCO_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle',
    'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
    'cat', 'dog', 'horse', 'sheep', 'cow', # ...총 91개
]
```

## 더 깊이 알아보기

### Mask R-CNN의 탄생 비화

Mask R-CNN(2017)의 1저자 Kaiming He는 이미 ResNet(2015)으로 컴퓨터 비전의 역사를 바꾼 인물입니다. Mask R-CNN의 설계 철학은 "기존에 잘 동작하는 것을 건드리지 말고, **최소한의 수정만 가하자**"였습니다. Faster R-CNN에 마스크 브랜치 하나를 **병렬로** 추가한 것이 전부인데, 이렇게 심플한 설계로 COCO 인스턴스 세그멘테이션 벤치마크를 석권했습니다. 논문은 ICCV 2017 Best Paper를 수상했으며, 2025년 기준 20,000회 이상 인용되었습니다.

### 인스턴스 세그멘테이션의 평가 지표

인스턴스 세그멘테이션은 객체 탐지와 동일하게 **AP(Average Precision)**로 평가하지만, IoU를 **마스크 기준**으로 계산합니다.

- **AP^mask**: 마스크 IoU 기준 Average Precision
- **AP50**, **AP75**: IoU 임계값 0.5, 0.75에서의 AP
- **AP_S, AP_M, AP_L**: 작은/중간/큰 객체별 AP

마스크 IoU는 바운딩 박스 IoU보다 훨씬 엄격합니다. 박스가 잘 맞아도 마스크 경계가 부정확하면 IoU가 크게 떨어지거든요.

## 흔한 오해와 팁

> ⚠️ **흔한 오해**: "Mask R-CNN은 느려서 실무에서 못 쓴다" — 원본 논문 시절에는 5 FPS 정도였지만, 최신 구현(detectron2, MMDetection)과 GPU 발전으로 **30+ FPS**도 가능합니다. 특히 TensorRT로 최적화하면 실시간 처리도 충분히 가능합니다.

> 🔥 **실무 팁**: Mask R-CNN을 커스텀 데이터에 학습할 때, **마스크 라벨링 도구** 선택이 중요합니다. CVAT, Label Studio, Roboflow 같은 도구가 폴리곤 기반 마스크 어노테이션을 지원합니다. COCO 포맷(폴리곤 좌표)으로 내보내면 대부분의 프레임워크에서 바로 사용할 수 있습니다.

> 💡 **알고 계셨나요?**: Mask R-CNN의 마스크 헤드는 28×28이라는 놀라울 정도로 작은 해상도를 사용합니다. 이걸 원본 크기로 업샘플링해도 충분히 좋은 마스크가 나오는데, 이는 대부분의 객체 윤곽이 **상대적으로 단순한 형태**이기 때문입니다. 물론 머리카락이나 나뭇잎 같은 복잡한 경계에서는 한계가 있습니다.

## 핵심 정리

| 개념 | 설명 |
|------|------|
| 인스턴스 세그멘테이션 | 각 객체를 개별적으로 구분하면서 픽셀 단위 마스크 예측 |
| Mask R-CNN | Faster R-CNN + 마스크 헤드 — Top-Down 방식의 대표 모델 |
| RoIAlign | 이중선형 보간으로 정확한 RoI 특징 추출 → 마스크 품질 향상 |
| YOLACT | 프로토타입 마스크 + 혼합 계수 → 실시간 인스턴스 세그멘테이션 |
| SOLOv2 | 박스 없이 동적 커널로 직접 마스크 예측 |
| AP^mask | 마스크 IoU 기준 Average Precision — 인스턴스 세그멘테이션 평가 지표 |

## 다음 섹션 미리보기

시맨틱 세그멘테이션은 **클래스는 구분하지만 개별 객체는 구분하지 못하고**, 인스턴스 세그멘테이션은 **개별 객체는 구분하지만 배경(stuff)은 다루지 못합니다**. 이 두 가지를 **하나로 통합**한 것이 바로 다음 섹션의 주제인 [파놉틱 세그멘테이션](./03-panoptic-segmentation.md)입니다. Mask2Former, OneFormer 등 하나의 모델로 모든 세그멘테이션을 통합하는 최신 접근법을 살펴봅니다.

## 참고 자료

- [Mask R-CNN (He et al., 2017)](https://arxiv.org/abs/1703.06870) - ICCV 2017 Best Paper, 인스턴스 세그멘테이션의 기준점
- [YOLACT: Real-time Instance Segmentation (Bolya et al., 2019)](https://arxiv.org/abs/1904.02689) - 실시간 인스턴스 세그멘테이션의 개척자
- [SOLOv2: Dynamic and Fast Instance Segmentation (Wang et al., 2020)](https://arxiv.org/abs/2003.10152) - Box-Free 인스턴스 세그멘테이션
- [Detectron2 공식 문서](https://detectron2.readthedocs.io/) - Meta의 객체 탐지/세그멘테이션 프레임워크
- [torchvision Instance Segmentation](https://pytorch.org/vision/stable/models.html#instance-segmentation) - PyTorch 공식 사전학습 모델
