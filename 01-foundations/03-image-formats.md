# 이미지 형식과 압축

> JPEG, PNG, WebP 등 형식별 특성과 압축 원리

## 개요

스마트폰으로 찍은 사진은 원본 그대로라면 수십 MB에 달합니다. 하지만 실제로 저장된 JPEG 파일은 1~3MB 수준입니다. 이 차이를 만드는 것이 바로 **이미지 압축**입니다. 이 섹션에서는 주요 이미지 형식의 차이와 압축이 동작하는 원리를 알아봅니다.

**선수 지식**: [이미지란 무엇인가](./01-what-is-image.md) — 픽셀, 해상도, 비트 깊이 / [색상 공간의 이해](./02-color-spaces.md) — RGB, YCbCr
**학습 목표**:
- 손실 압축과 무손실 압축의 차이를 설명할 수 있다
- JPEG, PNG, WebP 등 형식의 특성과 적합한 용도를 구분할 수 있다
- JPEG 압축의 기본 원리(DCT, 양자화)를 이해한다

## 왜 알아야 할까?

컴퓨터 비전에서 이미지 형식을 아는 것은 **데이터 품질**과 직결됩니다. JPEG로 반복 저장하면 눈에 보이지 않던 열화가 쌓여 모델 성능에 영향을 줄 수 있고, 투명 배경이 필요한데 JPEG를 쓰면 알파 채널이 사라집니다. 학습 데이터의 형식 선택은 모델 성능의 시작점입니다.

## 핵심 개념

### 1. 손실 압축 vs 무손실 압축

> 💡 **비유**: 긴 소설을 친구에게 전달한다고 생각해 보세요. **무손실 압축**은 소설을 **빈틈없이 압축 포장**해서 보내는 것 — 풀면 원본 그대로입니다. **손실 압축**은 소설을 **요약본으로 줄여** 보내는 것 — 핵심은 있지만 세부 묘사는 일부 빠집니다.

| 구분 | 손실 압축 (Lossy) | 무손실 압축 (Lossless) |
|------|-----------------|---------------------|
| 원본 복원 | 불가능 (일부 정보 손실) | 완벽 복원 가능 |
| 파일 크기 | 매우 작음 | 상대적으로 큼 |
| 대표 형식 | JPEG, WebP(손실) | PNG, BMP, WebP(무손실) |
| 적합한 용도 | 사진, 웹 이미지 | 다이어그램, 의료 영상, 편집용 |

### 2. JPEG — 사진의 표준

> 💡 **비유**: JPEG는 **눈에 안 띄는 부분을 슬쩍 지우는 화가**와 같습니다. 먼 하늘의 미세한 색 변화? 사람 눈은 어차피 구분 못하니 과감히 단순화합니다. 결과적으로 파일은 작아지고, 사진은 거의 그대로 보입니다.

**JPEG 압축이 동작하는 순서:**

1. **색상 변환**: RGB → YCbCr (밝기와 색을 분리)
2. **다운샘플링**: 색 채널(Cb, Cr)을 절반으로 줄임 (눈은 밝기에 민감, 색에는 둔감)
3. **DCT 변환**: 8×8 블록마다 주파수 성분으로 분해
4. **양자화**: 사람이 잘 못 느끼는 고주파 성분을 버림 ← **여기서 손실 발생!**
5. **엔트로피 코딩**: 남은 데이터를 효율적으로 압축

> DCT(이산 코사인 변환)란? 이미지의 한 블록을 "천천히 변하는 부분(저주파)"과 "급격히 변하는 부분(고주파)"으로 나누는 수학적 도구입니다. 사람 눈은 저주파에 민감하므로, 고주파를 버려도 품질 차이를 잘 느끼지 못합니다.

**JPEG 특성 요약:**

| 항목 | 내용 |
|------|------|
| 압축 방식 | 손실 압축 |
| 투명도 | 지원 안 함 |
| 색 깊이 | 24비트 (8비트 × 3채널) |
| 장점 | 파일 크기 매우 작음, 거의 모든 곳에서 지원 |
| 단점 | 저장할 때마다 품질 저하, 선명한 경계가 번짐 |
| 적합 | 사진, 자연 이미지 |
| 부적합 | 텍스트, 다이어그램, 로고 |

### 3. PNG — 품질을 지키는 수호자

> 💡 **비유**: PNG는 **한 글자도 빠뜨리지 않는 복사기**입니다. 원본 그대로를 보존하면서도, 반복되는 패턴을 찾아 효율적으로 저장합니다.

PNG는 **무손실 압축**을 사용합니다. 몇 번을 저장해도 화질이 절대 떨어지지 않습니다. 또한 **알파 채널(투명도)**을 지원하여 배경 없는 이미지를 만들 수 있습니다.

| 항목 | 내용 |
|------|------|
| 압축 방식 | 무손실 압축 (Deflate 알고리즘) |
| 투명도 | 지원 (알파 채널) |
| 색 깊이 | 최대 48비트 (16비트 × 3채널) |
| 장점 | 품질 손실 없음, 투명도 지원, 선명한 경계 |
| 단점 | 사진에서 JPEG보다 2~5배 큰 파일 |
| 적합 | 다이어그램, 로고, 스크린샷, 텍스트 포함 이미지 |
| 부적합 | 용량이 중요한 대량 사진 |

### 4. WebP — 둘 다 잘하는 후발주자

> 💡 **비유**: WebP는 **요약도 잘하고 원본 보존도 잘하는 만능 비서**입니다. 손실/무손실 압축을 모두 지원하면서, JPEG보다 25~35% 작고, PNG보다 26% 작습니다.

Google이 개발한 차세대 이미지 형식으로, 2025년 기준 주요 브라우저의 **97% 이상**이 지원합니다.

| 항목 | 내용 |
|------|------|
| 압축 방식 | 손실 + 무손실 모두 지원 |
| 투명도 | 지원 (알파 채널) |
| 애니메이션 | 지원 |
| 장점 | JPEG보다 작고, PNG보다 작음, 투명도+애니 가능 |
| 단점 | 일부 오래된 소프트웨어에서 미지원 |

### 5. 기타 형식

| 형식 | 압축 | 특징 | 주요 용도 |
|------|------|------|----------|
| **BMP** | 없음(비압축) | 원본 그대로, 매우 큰 파일 | 윈도우 기본 형식, 편집용 |
| **TIFF** | 무손실 | 레이어, 다중 페이지 지원 | 인쇄, 의료 영상, 위성 사진 |
| **GIF** | 무손실 | 256색 제한, 애니메이션 | 짧은 애니메이션, 아이콘 |
| **AVIF** | 손실+무손실 | WebP보다 더 높은 압축률 | 차세대 웹 이미지 (2025년~) |

### 6. 형식별 파일 크기 비교

같은 1920×1080 사진을 각 형식으로 저장하면:

| 형식 | 대략적 크기 | 원본 대비 |
|------|-----------|----------|
| BMP (비압축) | ~5.9 MB | 100% |
| PNG (무손실) | ~2.5 MB | ~42% |
| JPEG (품질 85) | ~0.5 MB | ~8.5% |
| WebP (품질 85) | ~0.35 MB | ~6% |
| AVIF (품질 85) | ~0.25 MB | ~4% |

> 비압축 원본 대비 JPEG는 약 **12배**, WebP는 약 **17배** 작아집니다.

## 실습: 직접 해보기

### 형식별 저장과 크기 비교

```python
import cv2
import os
import numpy as np

# 테스트용 이미지 생성 (또는 cv2.imread로 실제 이미지 사용)
img = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)

# 다양한 형식으로 저장
cv2.imwrite("test.bmp", img)
cv2.imwrite("test.png", img)
cv2.imwrite("test.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
cv2.imwrite("test.webp", img, [cv2.IMWRITE_WEBP_QUALITY, 85])

# 파일 크기 비교
for ext in ["bmp", "png", "jpg", "webp"]:
    size = os.path.getsize(f"test.{ext}")
    print(f"{ext.upper():>5}: {size:>10,} bytes ({size/1024/1024:.2f} MB)")
```

### JPEG 품질에 따른 차이 확인

```python
import cv2
import os

img = cv2.imread("sample.jpg")

# 다양한 품질로 JPEG 저장
for quality in [10, 30, 50, 70, 85, 95, 100]:
    filename = f"quality_{quality}.jpg"
    cv2.imwrite(filename, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    size = os.path.getsize(filename)
    print(f"품질 {quality:3d}: {size:>10,} bytes ({size/1024:.1f} KB)")
```

### 손실 압축의 누적 효과 확인

```python
import cv2
import numpy as np

# 원본 이미지 읽기
original = cv2.imread("sample.jpg")

# JPEG를 반복 저장하면 품질이 얼마나 떨어질까?
img = original.copy()
for i in range(50):
    cv2.imwrite("temp.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    img = cv2.imread("temp.jpg")

# 원본과 50회 재압축 이미지의 차이 계산
diff = cv2.absdiff(original, img)
mean_diff = np.mean(diff)
max_diff = np.max(diff)
print(f"50회 재압축 후:")
print(f"  평균 픽셀 차이: {mean_diff:.2f}")
print(f"  최대 픽셀 차이: {max_diff}")
```

## 더 깊이 알아보기

### CV 프로젝트에서의 형식 선택 가이드

| 상황 | 추천 형식 | 이유 |
|------|----------|------|
| 학습 데이터 원본 보관 | PNG 또는 TIFF | 품질 손실 방지 |
| 학습 데이터 대량 저장 | JPEG (품질 95) | 용량 절약 + 높은 품질 유지 |
| 데이터 증강 중간 결과 | NumPy (.npy) | 형식 변환 없이 배열 그대로 저장 |
| 웹/앱에 결과 표시 | WebP | 작은 용량 + 높은 품질 |
| 투명 배경 필요 | PNG | 알파 채널 지원 |

### NumPy 배열로 직접 저장하기

이미지 형식의 압축/변환 과정이 불필요할 때, NumPy 배열을 그대로 저장할 수 있습니다.

```python
import numpy as np
import cv2

img = cv2.imread("sample.jpg")

# NumPy 배열로 저장 (무손실, 빠름)
np.save("image.npy", img)

# 다시 불러오기
loaded = np.load("image.npy")
print(f"원본과 동일한가? {np.array_equal(img, loaded)}")  # True
```

## 핵심 정리

| 개념 | 설명 |
|------|------|
| **손실 압축** | 사람 눈에 안 띄는 정보를 버려 크기를 줄임 (JPEG) |
| **무손실 압축** | 원본 100% 보존하면서 크기를 줄임 (PNG) |
| **JPEG** | 사진에 최적, 작은 용량, 반복 저장 시 열화 |
| **PNG** | 품질 보존, 투명도 지원, 다이어그램에 적합 |
| **WebP** | 손실+무손실 모두 가능, JPEG/PNG보다 작음 |
| **DCT+양자화** | JPEG 압축 핵심 — 고주파(세밀한 변화)를 버려 용량 축소 |

## 다음 섹션 미리보기

이것으로 **Chapter 01: 이미지의 이해**가 마무리됩니다! 픽셀, 색상 공간, 이미지 형식까지 디지털 이미지의 기초를 쌓았습니다. 다음 챕터 **[OpenCV 시작하기](../02-classical-cv/01-opencv-basics.md)**에서는 이 지식을 바탕으로 실제 이미지를 읽고, 변환하고, 처리하는 실전 기술을 배웁니다.

## 참고 자료

- [MDN - Image file type and format guide](https://developer.mozilla.org/en-US/docs/Web/Media/Guides/Formats/Image_types) - 웹 이미지 형식 종합 가이드
- [Google Developers - WebP Compression Study](https://developers.google.com/speed/webp/docs/webp_study) - WebP vs JPEG 공식 비교 연구
- [Understanding DCT and Quantization in JPEG (DEV Community)](https://dev.to/marycheung021213/understanding-dct-and-quantization-in-jpeg-compression-1col) - JPEG 압축의 DCT와 양자화를 초보자용으로 설명
- [GeeksforGeeks - Process of JPEG Data Compression](https://www.geeksforgeeks.org/computer-graphics/process-of-jpeg-data-compression/) - JPEG 압축 5단계 상세 설명
- [Best Web Image Format for 2026](https://www.thecssagency.com/blog/best-web-image-format) - 최신 이미지 형식 트렌드 비교
