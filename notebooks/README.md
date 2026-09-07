# 논문/모델 실습 Notebook 로드맵

기존 `lessons/`는 MNIST 기반 MLP/CNN 기초 실습으로 유지하고, 그 다음 단계는 이 폴더에서 진행합니다.

이 폴더의 목표는 완성 코드를 빠르게 실행하는 것이 아니라, **코드를 한 cell씩 직접 타이핑하고 중간 결과를 시각화하면서 모델의 데이터 흐름을 이해하는 것**입니다.

## 권장 순서

| 단계 | 폴더 | 핵심 질문 |
|---:|---|---|
| 1 | `01_mlp/` | 이미지를 vector로 보면 무엇을 잃는가? |
| 2 | `02_cnn/` | convolution filter와 feature map은 무엇을 학습하는가? |
| 3 | `03_resnet/` | residual connection은 왜 깊은 모델 학습에 도움이 되는가? |
| 4 | `04_unet/` | encoder feature를 decoder에 다시 전달하는 이유는 무엇인가? |
| 5 | `05_attention/` | Q/K/V와 attention weight는 실제로 무엇을 계산하는가? |
| 6 | `06_transformer/` | attention block이 어떻게 Transformer encoder가 되는가? |
| 7 | `07_vit/` | 이미지를 patch token으로 바꾸면 CNN과 무엇이 달라지는가? |
| 8 | `08_sam/` | prompt가 segmentation mask를 어떻게 바꾸는가? |
| 9 | `09_sam2/` | image segmentation이 video memory/propagation으로 어떻게 확장되는가? |

## 학습 방식

MLP~ViT는 작은 모델을 직접 구현합니다. SAM/SAM2는 모델 규모가 크기 때문에 공식 pretrained 모델을 사용해 inference부터 시작하고, 그 다음 내부 모듈을 읽습니다.

각 notebook은 아래 흐름을 따릅니다.

```text
핵심 질문
→ 예상/가설
→ 입력/shape 확인
→ 작은 모듈 구현
→ 전체 모델 연결
→ 중간 tensor 확인
→ 학습/inference
→ 시각화
→ 실패 사례 확인
→ 결과 해석
```

## 시각화 원칙

가능한 한 각 모델의 내부를 눈으로 확인합니다.

- MLP: 입력, flatten, 첫 layer weight, confusion matrix
- CNN: filter, feature map, activation
- ResNet: `x`, `F(x)`, `F(x)+x`, Plain CNN과 비교
- U-Net: input / GT mask / prediction / overlay / error map
- Attention: score matrix, softmax, attention heatmap
- Transformer: head별 attention, layer별 shape
- ViT: patch 분할, patch embedding, attention map
- SAM: prompt와 mask overlay
- SAM2: frame별 mask propagation

중요한 결과는 `outputs/`에도 저장합니다.

## 논문 PDF

- U-Net: https://arxiv.org/pdf/1505.04597
- ResNet: https://arxiv.org/pdf/1512.03385
- Attention Is All You Need: https://arxiv.org/pdf/1706.03762
- ViT: https://arxiv.org/pdf/2010.11929
- SAM: https://arxiv.org/pdf/2304.02643
- SAM2: https://arxiv.org/pdf/2408.00714

## 각 notebook 완료 기준

```text
[ ] 핵심 질문을 설명할 수 있다.
[ ] 주요 tensor shape을 설명할 수 있다.
[ ] 핵심 연산을 직접 구현해봤다.
[ ] 최소 하나 이상의 내부 표현을 시각화했다.
[ ] 결과가 예상과 같은지 비교했다.
[ ] 실패 사례를 확인했다.
[ ] 이전 모델과의 차이를 한 문장으로 설명할 수 있다.
```
