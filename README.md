# PyTorch Deep Learning Practice

이 저장소는 **PyTorch 코드를 직접 타이핑하면서 딥러닝 구조와 데이터 흐름을 이해하기 위한 실습 저장소**입니다.

핵심 목표는 완성 코드를 빠르게 실행하는 것이 아니라,

- 각 모델이 왜 필요한지 이해하고
- 입력 tensor가 layer를 지나며 어떻게 변하는지 확인하고
- 핵심 모듈을 작은 형태로 직접 구현하고
- feature map / attention / segmentation 결과를 시각화하고
- 실패 사례와 학습 결과를 직접 해석하는 것

입니다.

## 학습 방식

저장소에서 직접 만드는 학습 실습은 기본적으로 `.py` 파일을 사용합니다.

```text
핵심 질문
   ↓
예상 / 가설
   ↓
입력 / shape 확인
   ↓
작은 모듈 직접 구현
   ↓
전체 구조 연결
   ↓
학습 또는 inference
   ↓
시각화 / 실패 사례
   ↓
결과 해석
```

`Jupyter Notebook(.ipynb)`을 기본 실습 형식으로 사용하지 않습니다. Notebook은 셀 실행 순서에 따라 이전 tensor나 model state가 남을 수 있으므로, 비교 실험은 위에서 아래로 한 번 실행되는 Python script를 우선합니다.

단, 외부 공개 저장소의 **공식 코드가 notebook 형태로 제공되는 경우에는 원본 형식을 억지로 변경하지 않습니다.** 그런 파일은 `external/` 등의 공식 소스 영역에서 그대로 보존하고, 필요한 학습 실습만 별도의 `.py` 파일로 작성합니다.

---

## 디렉터리 구조

```text
pytorch-deep-learning-practice/
├── lessons/          # 기존 MLP/CNN 기초 Python 실습
├── practice/         # 논문/모델별 직접 타이핑 실습
├── outputs/          # 그림, 로그, 모델 등 결과물
├── scripts/          # 환경 설정/점검 스크립트
├── external/         # 필요 시 clone되는 공식 외부 코드
├── AGENTS.md
├── README.md
└── environment.yml
```

현재 `practice/` 구조:

```text
practice/
├── 01_mlp/
│   └── mnist_mlp.py
├── 02_cnn/
│   ├── mnist_cnn.py
│   └── feature_maps.py
├── 03_resnet/
│   ├── residual_block.py
│   └── resnet18_cifar10.py
├── 04_unet/
│   ├── unet_architecture.py
│   └── segmentation.py
├── 05_attention/
│   ├── single_head_attention.py
│   └── multi_head_attention.py
├── 06_transformer/
│   └── transformer_encoder.py
├── 07_vit/
│   ├── patch_embedding.py
│   └── vit_cifar10.py
├── 08_sam/
│   ├── sam_image.py
│   └── sam_food.py
└── 09_sam2/
    ├── sam2_image.py
    └── sam2_video.py
```

각 파일은 완성 답안을 미리 넣기보다 **TODO와 최소 시작점만 제공**하고, 공부하면서 직접 채워가는 방식입니다.

---

## 권장 학습 순서

| 순서 | 모델 | 핵심 질문 |
|---:|---|---|
| 1 | MLP | 이미지를 vector로 펼치면 무엇을 잃는가? |
| 2 | CNN | convolution은 spatial structure를 어떻게 활용하는가? |
| 3 | ResNet | `F(x) + x`가 깊은 모델 학습에 왜 도움이 되는가? |
| 4 | U-Net | encoder feature를 decoder에 다시 전달하는 이유는 무엇인가? |
| 5 | Attention | Q/K/V와 attention weight는 실제로 무엇을 계산하는가? |
| 6 | Transformer | Attention + FFN + Residual + LayerNorm은 어떻게 block을 이루는가? |
| 7 | ViT | 이미지를 patch token으로 바꾸면 CNN과 무엇이 달라지는가? |
| 8 | SAM | point/box prompt가 mask를 어떻게 바꾸는가? |
| 9 | SAM2 | 한 frame의 prompt가 memory를 통해 다음 frame으로 어떻게 전달되는가? |

현재 논문 실습은 **ResNet**부터 진행합니다.

---

## 환경 설정

VSCode + Miniconda/Anaconda 기준입니다.

저장소 clone:

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
```

환경 생성/업데이트:

```bash
bash scripts/setup_env.sh
```

설정 후:

```bash
conda activate pytorch-dl-practice
python scripts/00_check_environment.py
```

실습 실행 예:

```bash
python practice/03_resnet/residual_block.py
```

VSCode에서는 `.py` 파일을 열고 필요한 코드를 직접 타이핑한 뒤 터미널에서 실행하면 됩니다.

---

## 기존 기초 실습

기존 `lessons/`는 유지합니다.

```text
lessons/
├── 01_tensor_basics.py
├── 02_mnist_data.py
├── 03_mnist_mlp.py
├── 04_mnist_cnn.py
└── 05_mnist_cnn_feature_maps.py
```

예:

```bash
python lessons/03_mnist_mlp.py
python lessons/04_mnist_cnn.py
python lessons/05_mnist_cnn_feature_maps.py
```

---

## 시각화 원칙

accuracy 하나만 확인하지 않습니다.

| 모델 | 주요 관찰 대상 |
|---|---|
| MLP | 입력, flatten, 첫 layer weight, confusion matrix |
| CNN | filter, feature map, activation |
| ResNet | `x`, `F(x)`, `F(x)+x`, Plain CNN과 학습 비교 |
| U-Net | input, GT mask, prediction, overlay, error map |
| Attention | Q/K/V shape, score matrix, softmax, attention heatmap |
| Transformer | residual 전후, LayerNorm, head별 attention |
| ViT | patch 분할, token shape, positional embedding, attention map |
| SAM | prompt와 mask overlay |
| SAM2 | frame별 mask propagation과 memory 효과 |

중요한 결과는 `outputs/`에도 저장합니다.

예:

```text
outputs/03_resnet/plain_vs_resnet_loss.png
outputs/04_unet/prediction_overlay.png
outputs/05_attention/attention_heatmap.png
outputs/07_vit/image_patches.png
```

---

## 참고 논문

| 모델/논문 | PDF |
|---|---|
| ResNet — *Deep Residual Learning for Image Recognition* | https://arxiv.org/pdf/1512.03385 |
| U-Net — *Convolutional Networks for Biomedical Image Segmentation* | https://arxiv.org/pdf/1505.04597 |
| Transformer — *Attention Is All You Need* | https://arxiv.org/pdf/1706.03762 |
| ViT — *An Image is Worth 16×16 Words* | https://arxiv.org/pdf/2010.11929 |
| SAM — *Segment Anything* | https://arxiv.org/pdf/2304.02643 |
| SAM2 — *SAM 2: Segment Anything in Images and Videos* | https://arxiv.org/pdf/2408.00714 |

---

## SAM / SAM2

SAM과 SAM2는 처음부터 재구현하지 않고 공식 pretrained model과 공식 source를 사용해 inference 및 내부 구조를 분석합니다.

SAM 추가 설정:

```bash
bash scripts/setup_sam.sh
```

SAM2 추가 설정:

```bash
bash scripts/setup_sam2.sh
```

SAM2 공식 저장소가 `external/sam2/` 아래에 clone되면, 그 안의 공식 `.ipynb` 파일은 **원본 공개 코드이므로 그대로 보존**합니다.

우리 쪽 실습은 다음 Python 파일에서 진행합니다.

```text
practice/08_sam/sam_image.py
practice/08_sam/sam_food.py
practice/09_sam2/sam2_image.py
practice/09_sam2/sam2_video.py
```

---

## 실습 완료 체크리스트

```text
[ ] 핵심 질문을 내 말로 설명할 수 있는가?
[ ] 입력 tensor의 shape / dtype / range를 확인했는가?
[ ] 주요 layer의 출력 shape를 설명할 수 있는가?
[ ] 핵심 연산을 직접 타이핑해서 구현해봤는가?
[ ] 최소 하나 이상의 중간 표현을 시각화했는가?
[ ] 예상과 실제 결과를 비교했는가?
[ ] 실패 사례를 하나 이상 확인했는가?
[ ] 논문의 주장과 실습 결과를 구분해서 설명할 수 있는가?
```

코드가 실행된다는 것과 구조를 이해했다는 것은 같은 의미가 아닙니다.
