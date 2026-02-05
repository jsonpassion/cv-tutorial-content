# 손실 함수와 옵티마이저

> Cross-Entropy, MSE, Adam, SGD

## 개요

역전파에서 "손실을 줄이는 방향으로 가중치를 조정한다"고 배웠습니다. 여기서 두 가지 질문이 남습니다: **"손실"을 어떻게 측정하는가?** (손실 함수) 그리고 **"조정"을 구체적으로 어떻게 하는가?** (옵티마이저) 이 섹션에서 둘 다 다룹니다.

**선수 지식**: [역전파 알고리즘](./03-backpropagation.md) — 경사하강법, 기울기
**학습 목표**:
- 분류와 회귀에 적합한 손실 함수를 선택할 수 있다
- SGD, Adam 등 옵티마이저의 차이를 이해한다
- PyTorch에서 손실 함수와 옵티마이저를 조합하여 학습할 수 있다

## 왜 알아야 할까?

잘못된 손실 함수를 쓰면 모델이 엉뚱한 것을 최적화합니다. 부적절한 옵티마이저는 학습이 느리거나 불안정해집니다. **Cross-Entropy + Adam**은 현대 딥러닝의 표준 조합이지만, 상황에 따라 다른 선택이 필요할 때를 아는 것이 중요합니다.

## 핵심 개념

### 1. 손실 함수 — 얼마나 틀렸는지 측정

> 💡 **비유**: 손실 함수는 **채점표**입니다. 학생(모델)의 답안(예측)을 정답과 비교해 점수를 매깁니다. 다만 여기서 점수가 **낮을수록 좋습니다** — 틀린 정도를 측정하는 것이니까요.

### 2. MSE (Mean Squared Error) — 회귀 문제의 기본

> 💡 **비유**: 다트를 던질 때 **과녁 중심에서 얼마나 벗어났는지를 제곱**해서 측정합니다. 많이 벗어나면 벌점이 기하급수적으로 커집니다.

> **MSE = (1/N) × Σ(예측값 − 정답)²**

```python
import torch
import torch.nn as nn

criterion = nn.MSELoss()

prediction = torch.tensor([2.5, 3.0, 4.5])
target = torch.tensor([3.0, 3.0, 5.0])

loss = criterion(prediction, target)
print(f"예측: {prediction.tolist()}")
print(f"정답: {target.tolist()}")
print(f"MSE 손실: {loss.item():.4f}")
# (0.25 + 0.0 + 0.25) / 3 = 0.1667
```

| 특성 | 내용 |
|------|------|
| 용도 | **회귀** (연속 값 예측) |
| 범위 | 0 ~ ∞ (0이 최적) |
| 특징 | 큰 오차에 큰 벌점 (제곱 효과) |
| 예시 | 집값 예측, 온도 예측, 바운딩 박스 좌표 |

### 3. Cross-Entropy — 분류 문제의 표준

> 💡 **비유**: "이 사진은 고양이일까, 강아지일까?" 모델이 "고양이 90%, 강아지 10%"라고 답했는데 정답이 고양이라면 벌점이 적습니다. 하지만 "고양이 10%, 강아지 90%"라고 확신있게 틀리면 벌점이 **매우 크게** 부과됩니다.

```python
import torch
import torch.nn as nn

# 분류: 3개 클래스 (고양이, 강아지, 새)
criterion = nn.CrossEntropyLoss()

# 모델 출력 (logits, softmax 전 값)
logits = torch.tensor([[2.0, 1.0, 0.1]])  # 클래스0(고양이)에 높은 점수
target = torch.tensor([0])                 # 정답: 클래스0(고양이)

loss = criterion(logits, target)
print(f"모델 출력(logits): {logits.tolist()}")
print(f"정답 클래스: {target.item()}")
print(f"Cross-Entropy 손실: {loss.item():.4f}")

# 확률로 변환해서 확인
probs = torch.softmax(logits, dim=1)
print(f"확률: {[f'{p:.2%}' for p in probs[0].tolist()]}")
```

| 특성 | 내용 |
|------|------|
| 용도 | **분류** (클래스 예측) |
| 입력 | Logits (softmax 전 값) — PyTorch가 내부에서 softmax 적용 |
| 특징 | 확신있게 틀릴수록 큰 벌점 |
| 예시 | 이미지 분류, 감정 분석, 객체 탐지 클래스 |

### 4. 이진 분류용 — BCELoss

```python
import torch
import torch.nn as nn

# 이진 분류: 스팸(1) vs 정상(0)
criterion = nn.BCEWithLogitsLoss()  # Sigmoid + BCE를 합친 버전

logits = torch.tensor([2.0, -1.0, 0.5])   # 모델 출력
target = torch.tensor([1.0, 0.0, 1.0])     # 정답

loss = criterion(logits, target)
print(f"BCE 손실: {loss.item():.4f}")
```

### 5. 손실 함수 선택 가이드

| 문제 유형 | 손실 함수 | PyTorch 클래스 |
|----------|----------|---------------|
| 회귀 (연속 값) | MSE | `nn.MSELoss()` |
| 회귀 (이상치에 강건) | MAE / Huber | `nn.L1Loss()` / `nn.SmoothL1Loss()` |
| 다중 분류 | Cross-Entropy | `nn.CrossEntropyLoss()` |
| 이진 분류 | Binary CE | `nn.BCEWithLogitsLoss()` |
| 세그멘테이션 | Dice + CE | 커스텀 조합 |

---

### 6. 옵티마이저 — 어떻게 업데이트할 것인가

> 💡 **비유**: 경사하강법이 "내려가는 방향으로 걸어라"라면, 옵티마이저는 **걷는 전략**입니다. SGD는 우직하게 한 방향, Adam은 지형을 기억하며 영리하게 걷습니다.

### 7. SGD (Stochastic Gradient Descent)

가장 기본적인 옵티마이저. 기울기 방향으로 학습률만큼 이동합니다.

```python
import torch.optim as optim

optimizer = optim.SGD(model.parameters(), lr=0.01)

# 모멘텀 추가 (관성을 주어 진동 줄임)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 8. Adam — 현대의 기본값

Adam = **Ada**ptive **M**oment Estimation. 각 파라미터마다 학습률을 자동 조절합니다.

> 💡 **비유**: SGD가 모든 길을 같은 보폭으로 걷는다면, Adam은 **평탄한 길에서는 빠르게, 가파른 길에서는 조심스럽게** 걸음폭을 자동 조절합니다.

```python
import torch.optim as optim

# Adam (가장 많이 사용)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# AdamW (가중치 감쇠 개선 버전, Transformer에서 표준)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

### 9. 옵티마이저 비교

| 옵티마이저 | 학습률 조절 | 모멘텀 | 장점 | 단점 |
|-----------|-----------|--------|------|------|
| **SGD** | 고정 | 선택 | 일반화 성능 좋음 | 수렴 느림, 튜닝 어려움 |
| **SGD+Momentum** | 고정 | 있음 | 진동 감소, 더 빠름 | 학습률 수동 설정 |
| **Adam** | 파라미터별 자동 | 있음 | 빠른 수렴, 튜닝 쉬움 | 가끔 일반화 약함 |
| **AdamW** | 파라미터별 자동 | 있음 | Adam + 올바른 가중치 감쇠 | **Transformer 표준** |

> **실무 가이드**: 처음에는 **Adam(lr=0.001)**로 시작, 미세 조정이 필요하면 **AdamW** 또는 **SGD+Momentum**을 시도하세요.

## 실습: 직접 해보기

### 옵티마이저별 학습 속도 비교

```python
import torch
import torch.nn as nn

def train_with_optimizer(opt_name, opt_class, lr):
    """동일한 모델을 다른 옵티마이저로 학습하고 손실 변화 관찰"""
    torch.manual_seed(42)

    model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
    criterion = nn.MSELoss()

    if opt_name == "SGD+Momentum":
        optimizer = opt_class(model.parameters(), lr=lr, momentum=0.9)
    else:
        optimizer = opt_class(model.parameters(), lr=lr)

    x = torch.linspace(-3, 3, 50).unsqueeze(1)
    y = torch.sin(x)

    losses = []
    for epoch in range(200):
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            losses.append(loss.item())

    print(f"{opt_name:15s} | 손실 변화: {' → '.join(f'{l:.4f}' for l in losses)}")

train_with_optimizer("SGD", torch.optim.SGD, 0.01)
train_with_optimizer("SGD+Momentum", torch.optim.SGD, 0.01)
train_with_optimizer("Adam", torch.optim.Adam, 0.001)
train_with_optimizer("AdamW", torch.optim.AdamW, 0.001)
```

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **MSE** | 회귀용 손실. 오차의 제곱 평균 |
| **Cross-Entropy** | 분류용 손실. 확신있게 틀리면 큰 벌점 |
| **SGD** | 기본 경사하강법. 단순하지만 일반화 좋음 |
| **Adam** | 적응적 학습률. 빠른 수렴, 가장 널리 사용 |
| **AdamW** | Adam + 가중치 감쇠 개선. Transformer의 표준 |
| **학습률** | 가장 중요한 하이퍼파라미터. 보통 0.001에서 시작 |

## 다음 섹션 미리보기

신경망의 이론적 기초를 모두 다뤘습니다. 다음 섹션 **[PyTorch 기초](./05-pytorch-fundamentals.md)**에서는 지금까지 배운 개념들을 PyTorch로 실전 구현하는 방법을 체계적으로 정리합니다. 텐서 연산, Dataset, DataLoader, 학습 루프까지 실무에 필요한 모든 것을 다룹니다.

## 참고 자료

- [DataCamp - Cross-Entropy Loss Function](https://www.datacamp.com/tutorial/the-cross-entropy-loss-function-in-machine-learning) - Cross-Entropy의 수학적 배경과 실용적 사용법
- [Machine Learning Mastery - How to Choose Loss Functions](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/) - 문제 유형별 손실 함수 선택 가이드
- [Medium - Demystifying Loss Functions & Adam Optimizer](https://medium.com/@akankshasinha247/how-ai-learns-demystifying-loss-functions-the-adam-optimizer-ed29862e389c) - 손실 함수와 Adam의 직관적 설명
- [arXiv - Loss Functions and Metrics in Deep Learning](https://arxiv.org/html/2307.02694v5) - 손실 함수 종합 서베이 논문
