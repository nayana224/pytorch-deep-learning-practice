# 04. U-Net 실습

이 폴더의 목표는 U-Net 완성 코드를 복사하는 것이 아니라, **논문에서 실제 사용한 데이터와 구조를 가능한 한 그대로 따라가며 PyTorch로 직접 타이핑하고 이해하는 것**이다.

## 학습 방식

1. 논문이 사용한 데이터셋과 GT 구조를 먼저 확인한다.
2. ChatGPT가 현재 단계에 필요한 작은 실제 코드를 제시한다.
3. 사용자가 코드를 직접 타이핑한다.
4. 실행 전에 tensor shape과 데이터 의미를 예상한다.
5. 직접 실행하고 결과를 확인한다.
6. 이해되지 않는 코드, shape 변화, 논문과의 대응을 질문한다.
7. 이해가 끝나면 다음 코드 조각으로 넘어간다.

전체 완성 코드를 한 번에 받지 않는다.

## 1. 기준 데이터셋

첫 번째 기준은 논문의 **ISBI 2012 EM segmentation challenge** 데이터다.

논문에 따르면:
- training image: 30장
- image size: 512x512
- serial section transmission electron microscopy 이미지
- 각 training image에 fully annotated segmentation map이 존재
- GT는 cell과 membrane 구조를 구분하는 segmentation map
- test GT는 공개되지 않고 challenge server에서 평가
- 논문 평가 지표는 warping error, Rand error, pixel error

따라서 synthetic data는 기본 실습 경로에서 제외하고, 실제 EM image / GT mask를 먼저 다룬다. synthetic data는 데이터 로딩 문제와 모델 문제를 분리할 필요가 있을 때만 진단용으로 사용한다.

## 2. `unet_architecture.py`

논문 Figure 1의 original U-Net 구조를 PyTorch로 직접 따라간다.

```text
input tile
→ valid 3x3 convolution + ReLU
→ valid 3x3 convolution + ReLU
→ max pooling
→ contracting path
→ bottleneck
→ up-convolution
→ encoder feature crop
→ concatenation
→ expanding path
→ 1x1 convolution
→ segmentation logits
```

논문 Figure 1에서는 572x572 input tile에서 시작해 388x388 output segmentation map을 만든다. 실제 dataset image 크기 512x512와 네트워크 input tile 크기는 같은 개념이 아니므로, 데이터 로딩과 tile 생성 과정을 따로 확인한다.

핵심 관찰 항목:
- valid convolution에서 spatial size가 줄어드는 이유
- encoder의 channel 증가와 resolution 감소
- pooling 전 feature가 skip으로 전달되는 이유
- upsampled feature와 encoder feature의 spatial mismatch
- crop + concat의 실제 tensor shape
- 마지막 1x1 convolution의 역할

## 3. `segmentation.py`

논문 데이터로 학습 데이터 흐름을 추적한다.

```text
EM raw image / GT segmentation
→ tile / preprocessing / augmentation
→ Dataset / DataLoader
→ U-Net
→ logits
→ pixel-wise loss
→ backward / optimizer
→ prediction
→ evaluation / visualization
```

먼저 image / GT의 실제 파일 구조, shape, dtype, value range를 확인한다. 그 다음 논문 기본 학습 흐름을 연결하고, weighted loss와 elastic deformation은 별도 단계로 추가한다.

## 논문 재현과 학습 실습의 구분

가능한 한 논문 설정을 그대로 사용하지만, 논문에 충분히 명시되지 않은 부분이나 현재 환경에서 그대로 재현하기 어려운 부분은 임의로 숨기지 않는다. 무엇을 그대로 따랐고 무엇을 단순화했는지 기록한다.

## 완료 기준

1. Problem: sliding-window CNN의 비효율과 localization/context trade-off
2. Core idea: contracting path + expanding path + skip feature
3. Method: valid conv / downsampling / upsampling / crop / concat / 1x1 conv
4. Input / GT / Output / Loss: 실제 EM image와 GT가 model/loss까지 흐르는 과정
5. Evidence: 논문 지표와 prediction 결과가 저자 주장과 어떻게 연결되는지
6. My observation: feature / prediction / failure case에서 직접 본 현상
