# PyTorch Deep Learning Practice

이 저장소는 **PyTorch로 딥러닝 구조를 직접 구현하고, 중간 tensor와 시각화를 보면서 이해하기 위한 실습 저장소**입니다.

목표는 최신 모델을 많이 실행하는 것이 아니라,

- 코드를 한 단계씩 직접 타이핑하고
- 각 layer의 tensor shape과 데이터 흐름을 확인하고
- filter / feature map / attention / segmentation 결과를 시각화하고
- 왜 이런 구조가 필요한지 설명할 수 있게 되는 것

입니다.

기존 `lessons/`에는 MNIST 기반 MLP/CNN 기초 실습이 있고, 그 다음 단계는 `notebooks/`에서 **ResNet → U-Net → Attention → Transformer → ViT → SAM → SAM2** 순서로 진행합니다.

---

## 1. 이 저장소의 공부 방식

각 실습은 다음 흐름으로 진행합니다.

```text
핵심 질문
   ↓
실습 전 예상/가설
   ↓
작은 코드부터 직접 구현
   ↓
중간 tensor shape 확인
   ↓
학습 또는 inference
   ↓
가능한 한 많이 시각화
   ↓
결과 해석
   ↓
다음 모델과 연결
```

중요한 원칙은 다음과 같습니다.

- 처음부터 완성 코드를 통째로 복사하지 않습니다.
- notebook을 짧은 cell로 나누고 순서대로 직접 타이핑합니다.
- accuracy만 확인하지 않습니다.
- 중요한 tensor는 shape뿐 아니라 가능하면 실제 값이나 그림으로 확인합니다.
- 비교 실험에서는 한 번에 한 조건만 바꿉니다.
- 결과를 보기 전에 가능하면 먼저 예상한 뒤 실제 결과와 비교합니다.

---

## 2. 전체 학습 순서

### Phase 1 — 기초

```text
MLP
 ↓
CNN
 ↓
CNN Filter / Feature Map 시각화
```

### Phase 2 — 논문/현대 Vision 구조

```text
ResNet
 ↓
U-Net
 ↓
Attention
 ↓
Transformer
 ↓
ViT
 ↓
SAM
 ↓
SAM2
```

권장 notebook 구조:

```text
notebooks/
├── 01_mlp/
│   └── mnist_mlp.ipynb
├── 02_cnn/
│   ├── mnist_cnn.ipynb
│   └── feature_maps.ipynb
├── 03_resnet/
│   ├── residual_block.ipynb
│   └── resnet18_cifar10.ipynb
├── 04_unet/
│   ├── unet_architecture.ipynb
│   └── segmentation.ipynb
├── 05_attention/
│   ├── single_head_attention.ipynb
│   └── multi_head_attention.ipynb
├── 06_transformer/
│   └── transformer_encoder.ipynb
├── 07_vit/
│   ├── patch_embedding.ipynb
│   └── vit_cifar10.ipynb
├── 08_sam/
│   ├── sam_image.ipynb
│   └── sam_food.ipynb
└── 09_sam2/
    ├── sam2_image.ipynb
    └── sam2_video.ipynb
```

각 notebook은 처음부터 정답 코드를 모두 넣기보다 아래 흐름으로 직접 채워갑니다.

```text
학습 목표
 → 입력 데이터 / shape 확인
 → 작은 모듈부터 구현
 → 전체 모델 연결
 → forward 중간 tensor 확인
 → 학습 또는 inference
 → 시각화
 → 실패 사례 확인
 → 결과 해석
```

---

## 3. 모델별로 무엇을 시각화할까?

| 모델 | 반드시 보고 싶은 것 |
|---|---|
| MLP | 입력 이미지, flatten 결과, 첫 layer weight, confusion matrix |
| CNN | convolution filter, training 전/후 feature map, prediction |
| ResNet | residual block의 `x`, `F(x)`, `F(x)+x`, Plain CNN과 loss 비교 |
| U-Net | input, ground truth mask, predicted mask, overlay, error map |
| Attention | Q/K/V shape, score matrix, softmax 결과, attention heatmap |
| Transformer | head별 attention, residual 전/후, layer output shape |
| ViT | patch 분할, patch embedding, positional embedding, attention map |
| SAM | point/box prompt와 mask overlay, prompt 변화에 따른 mask 차이 |
| SAM2 | frame별 mask propagation, prompt 이후 video tracking 결과 |

시각화 결과 중 중요한 것은 notebook 화면에서만 보지 않고 `outputs/`에도 저장합니다.

예:

```text
outputs/03_resnet/plain_vs_resnet_loss.png
outputs/04_unet/prediction_overlay.png
outputs/05_attention/attention_heatmap.png
outputs/07_vit/image_patches.png
```

---

## 4. 참고 논문

논문을 먼저 전부 이해한 뒤 코딩하려고 하기보다, **논문의 핵심 구조를 읽고 바로 작은 구현과 시각화로 확인하는 방식**을 권장합니다.

| 순서 | 모델/논문 | PDF |
|---:|---|---|
| 1 | U-Net — *Convolutional Networks for Biomedical Image Segmentation* (2015) | https://arxiv.org/pdf/1505.04597 |
| 2 | ResNet — *Deep Residual Learning for Image Recognition* (2015) | https://arxiv.org/pdf/1512.03385 |
| 3 | Transformer — *Attention Is All You Need* (2017) | https://arxiv.org/pdf/1706.03762 |
| 4 | ViT — *An Image is Worth 16×16 Words* (2020) | https://arxiv.org/pdf/2010.11929 |
| 5 | SAM — *Segment Anything* (2023) | https://arxiv.org/pdf/2304.02643 |
| 6 | SAM2 — *SAM 2: Segment Anything in Images and Videos* (2024) | https://arxiv.org/pdf/2408.00714 |

---

# 5. 가장 쉬운 환경 설정 방법

VSCode + Miniconda 기준입니다.

## 준비물

먼저 아래 두 가지가 있어야 합니다.

- VSCode
- Miniconda 또는 Anaconda

VSCode 확장은 아래 두 개를 설치하면 됩니다.

- Python
- Jupyter

---

## 방법 A — Notebook 실습용 환경 한 번에 만들기

저장소를 clone합니다.

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
```

그 다음 아래 명령 하나를 실행합니다.

```bash
bash scripts/setup_notebook_env.sh
```

이 스크립트는 `environment.notebook.yml`을 사용해서 다음 Conda 환경을 만듭니다.

```text
pytorch-dl-notebook
```

설치 후:

```bash
conda activate pytorch-dl-notebook
```

Jupyter kernel도 등록합니다.

```bash
python -m ipykernel install --user \
  --name pytorch-dl-notebook \
  --display-name "Python (pytorch-dl-notebook)"
```

환경 확인:

```bash
python scripts/00_check_environment.py
```

---

# 6. VSCode에서 `.ipynb` 실행하기

예를 들어:

```text
notebooks/03_resnet/residual_block.ipynb
```

파일을 엽니다.

VSCode 오른쪽 위의 **Kernel 선택** 버튼을 누른 다음:

```text
Python (pytorch-dl-notebook)
```

을 선택합니다.

이제 각 cell 왼쪽의 실행 버튼을 눌러 한 cell씩 확인하면 됩니다.

### interpreter가 안 보일 때

터미널에서 먼저:

```bash
conda activate pytorch-dl-notebook
python -m ipykernel install --user \
  --name pytorch-dl-notebook \
  --display-name "Python (pytorch-dl-notebook)"
```

을 다시 실행한 뒤 VSCode를 새로고침합니다.

---

# 7. GPU / CPU 확인

Notebook 첫 cell에서 아래 코드를 실행하면 됩니다.

```python
import torch

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

GPU가 없으면:

```text
CUDA available: False
```

라고 나와도 정상입니다.

MLP, CNN, ResNet, U-Net, Attention, Transformer, Tiny ViT 실습은 CPU에서도 가능합니다.

SAM/SAM2, 특히 SAM2 video 실습은 GPU 사용을 권장합니다.

---

# 8. 기존 Python script 실습

기존 `lessons/`는 그대로 유지합니다.

```text
lessons/
├── 01_tensor_basics.py
├── 02_mnist_data.py
├── 03_mnist_mlp.py
├── 04_mnist_cnn.py
└── 05_mnist_cnn_feature_maps.py
```

실행 예:

```bash
conda activate pytorch-dl-notebook
python lessons/03_mnist_mlp.py
python lessons/04_mnist_cnn.py
python lessons/05_mnist_cnn_feature_maps.py
```

처음에는 이 MLP/CNN script를 충분히 본 다음 `notebooks/03_resnet/`으로 넘어가는 것을 권장합니다.

---

# 9. 모델별 실습 목표

| 순서 | 모델 | 핵심 질문 |
|---:|---|---|
| 1 | MLP | 이미지를 vector로 펼치면 무엇을 잃는가? |
| 2 | CNN | convolution filter와 feature map은 무엇을 학습하는가? |
| 3 | ResNet | `F(x) + x`가 깊은 모델 학습에 왜 도움이 되는가? |
| 4 | U-Net | encoder feature를 decoder에 다시 전달하는 이유는 무엇인가? |
| 5 | Attention | Q/K/V와 attention weight는 실제로 무엇을 계산하는가? |
| 6 | Transformer | Attention + FFN + Residual + LayerNorm은 어떻게 block을 이루는가? |
| 7 | ViT | 이미지를 patch token으로 바꾸면 CNN과 무엇이 달라지는가? |
| 8 | SAM | point/box prompt가 segmentation mask를 어떻게 바꾸는가? |
| 9 | SAM2 | 한 frame의 prompt가 memory를 통해 다음 frame으로 어떻게 전달되는가? |

MLP~ViT는 작은 형태를 직접 구현합니다.

SAM/SAM2는 모델이 크므로 처음부터 재구현하지 않고 **공식 pretrained 모델 inference → 내부 source 분석** 순서로 진행합니다.

---

# 10. 각 notebook에서 반복할 체크리스트

실습 중에는 아래를 반복해서 확인합니다.

```text
[ ] 이 실습의 핵심 질문을 한 문장으로 설명할 수 있는가?
[ ] 입력 tensor의 shape / dtype / range를 확인했는가?
[ ] 각 주요 layer의 출력 shape를 설명할 수 있는가?
[ ] 학습되는 parameter와 단순 intermediate tensor를 구분할 수 있는가?
[ ] 최소 하나 이상의 중간 표현을 시각화했는가?
[ ] 예상과 실제 결과가 같은가?
[ ] 실패 사례를 하나 이상 확인했는가?
[ ] 이전 모델과 무엇이 달라졌는지 설명할 수 있는가?
```

코드가 실행된다는 것과 구조를 이해했다는 것은 같은 의미가 아닙니다.

---

# 11. SAM 추가 설정

기본 notebook 환경을 먼저 만든 상태에서 실행합니다.

```bash
bash scripts/setup_notebook_env.sh
conda activate pytorch-dl-notebook
```

SAM 설치:

```bash
bash scripts/setup_sam.sh
```

이 스크립트는 Meta 공식 `segment-anything` package를 설치합니다.

SAM checkpoint는 크기 때문에 저장소에 commit하지 않습니다.

권장 구조:

```text
checkpoints/
└── sam/
    └── ... .pth
```

`.pth`, `.pt`, `checkpoints/`는 `.gitignore`에 포함되어 있습니다.

SAM 실습:

```text
notebooks/08_sam/sam_image.ipynb
notebooks/08_sam/sam_food.ipynb
```

처음에는 일반 이미지에서 point/box prompt를 실험한 다음 실제 Custom Food 이미지로 넘어갑니다.

---

# 12. SAM2 추가 설정

SAM2는 기본 모델보다 dependency와 GPU 요구가 더 큽니다.

기본 notebook 환경을 만든 뒤:

```bash
conda activate pytorch-dl-notebook
bash scripts/setup_sam2.sh
```

스크립트가 Meta 공식 SAM2 저장소를 다음 위치에 clone합니다.

```text
external/sam2/
```

그리고 notebook dependency까지 editable install 합니다.

현재 Meta 공식 SAM2 README 기준으로 SAM2는 Python 3.10 이상, PyTorch 2.5.1 이상을 요구합니다. 이 저장소의 notebook 환경은 그보다 높은 PyTorch 버전을 사용하도록 구성되어 있습니다.

처음에는 큰 checkpoint보다 다음 작은 모델로 시작하는 것을 권장합니다.

```text
sam2.1_hiera_tiny
```

공식 checkpoint 다운로드 방법은 다음 파일에서도 확인할 수 있습니다.

```text
external/sam2/README.md
external/sam2/checkpoints/
```

SAM2 실습 순서:

```text
notebooks/09_sam2/sam2_image.ipynb
       ↓
notebooks/09_sam2/sam2_video.ipynb
```

먼저 image predictor를 이해한 뒤 video propagation으로 넘어갑니다.

---

# 13. 데이터와 checkpoint를 GitHub에 올리지 않는 이유

아래 항목은 기본적으로 Git에 포함하지 않습니다.

```text
data/
external/
checkpoints/
*.pth
*.pt
.ipynb_checkpoints/
```

이유:

- dataset 용량이 큼
- SAM/SAM2 checkpoint가 매우 큼
- 외부 공식 repository까지 현재 저장소에 중복 commit할 필요가 없음
- notebook 자동 cache는 학습 내용과 관계없음

---

# 14. 실습 원칙

이 저장소에서는 단순히 `accuracy가 높다`로 끝내지 않습니다.

### 항상 먼저 확인할 것

```text
input shape
↓
dtype / range
↓
model input/output shape
↓
중간 activation
↓
loss
↓
시각화
```

### 비교 실험

한 번에 한 조건만 바꿉니다.

예:

```text
Plain CNN vs Residual CNN
```

이라면 architecture의 residual connection 외에 가능한 한 같은 조건을 유지합니다.

### 결과를 보기 전에 가설을 먼저 적기

예:

> residual connection이 있으면 깊은 모델에서 gradient가 더 안정적으로 전달될 것으로 예상한다.

실행 후 예상과 다르면 그 차이를 분석합니다.

---

# 15. Custom Food와 연결

최종 목적은 모델 이름을 많이 구현하는 것이 아니라 배운 개념을 실제 perception 문제에 연결하는 것입니다.

예:

```text
U-Net
image → food segmentation mask

SAM
image + prompt → food mask

SAM2
video + prompt → temporal food mask
```

향후 `sam_food.ipynb`에서는 실제 Custom Food 이미지에서 다음을 비교합니다.

```text
현재 food mask
vs
SAM mask
vs
향후 learned segmentation
```

그리고 mask accuracy만 볼 것이 아니라 tray wall, sauce, reflection, depth invalid region 등 실제 manipulation pipeline에서 문제가 되는 failure case를 기록합니다.

새 모델을 넣는 것 자체를 목표로 하지 않습니다. 기존 pipeline의 어떤 실패를 해결하려는지 먼저 정의하고, 실제 개선되는지 비교 실험으로 확인합니다.

---

## 빠른 시작 요약

처음 clone한 뒤 아래 순서만 실행하면 됩니다.

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice

bash scripts/setup_notebook_env.sh
conda activate pytorch-dl-notebook

python -m ipykernel install --user \
  --name pytorch-dl-notebook \
  --display-name "Python (pytorch-dl-notebook)"

python scripts/00_check_environment.py
```

그 다음 VSCode에서:

```text
notebooks/01_mlp/mnist_mlp.ipynb
```

를 열고 Kernel을:

```text
Python (pytorch-dl-notebook)
```

으로 선택하면 됩니다.

그 이후에는 **코드를 한 셀씩 직접 타이핑하고, 바로 실행하고, 바로 시각화해서 확인하는 방식**으로 진행합니다.
