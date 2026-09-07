# 논문/모델 실습 Notebook 로드맵

기존 `lessons/`는 MNIST 기반 MLP/CNN 기초 실습으로 유지하고, 그 다음 단계는 이 폴더에서 진행합니다.

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

각 notebook은 `목표 → 입력/shape → 모델 → 중간 결과 → 시각화 → 해석` 순서로 작성합니다.
