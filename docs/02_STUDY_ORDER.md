# 02. 논문 실습 순서

이 문서는 `practice/`를 어떤 순서로 읽고 실행할지 정리한다.

## 공통 순서

각 논문은 아래 순서로 본다.

```text
1. README
2. raw data
3. model definition
4. shape / feature 확인
5. training 또는 inference
6. visualization / prediction
7. failure case
8. 논문 주장과 실습 관찰 비교
```

---

## 01. ResNet

논문: Deep Residual Learning for Image Recognition

데이터: CIFAR-10

먼저 읽을 파일:

```text
practice/01_resnet/README.md
practice/01_resnet/resnet.py
```

실행 순서:

```bash
python scripts/download_torchvision_data.py cifar10
python practice/01_resnet/01_data.py
python practice/01_resnet/02_model.py
python practice/01_resnet/03_train.py
python practice/01_resnet/04_analyze.py
```

핵심 확인:

```text
plain block vs residual block
F(x)
shortcut x
F(x) + x
16 → 32 → 64 channels
degradation problem
```

---

## 02. U-Net

논문: Convolutional Networks for Biomedical Image Segmentation

데이터: ISBI 2012 EM

먼저 읽을 파일:

```text
practice/02_unet/README.md
practice/02_unet/unet.py
```

실행 순서:

```bash
bash scripts/download_isbi2012.sh
python practice/02_unet/01_data.py
python practice/02_unet/02_model.py
python practice/02_unet/03_train.py --epochs 10
python practice/02_unet/04_analyze.py
```

핵심 확인:

```text
valid convolution
contracting path
up-convolution
center crop
copy + crop
channel concat
572 → 388
input / GT / logits / loss
```

---

## 03. DeepLabv3+

논문: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation

데이터: PASCAL VOC 2012

먼저 읽을 파일:

```text
practice/03_deeplabv3plus/README.md
practice/03_deeplabv3plus/deeplabv3plus.py
```

실행 순서:

```bash
bash scripts/download_voc2012.sh
python practice/03_deeplabv3plus/01_data.py
python practice/03_deeplabv3plus/02_atrous.py
python practice/03_deeplabv3plus/03_model.py
python practice/03_deeplabv3plus/04_train.py --epochs 20
python practice/03_deeplabv3plus/05_analyze.py
```

핵심 확인:

```text
atrous convolution
output stride
ASPP
low-level feature → 48 channels
concat
decoder refinement
mIoU
```

---

## 04. Vision Transformer

논문: An Image Is Worth 16x16 Words

데이터: CIFAR-100 (논문 downstream benchmark)

먼저 읽을 파일:

```text
practice/04_vit/README.md
practice/04_vit/vit.py
```

실행 순서:

```bash
python scripts/download_torchvision_data.py cifar100
python practice/04_vit/01_data.py
python practice/04_vit/02_patches.py
python practice/04_vit/03_model.py
python practice/04_vit/04_train.py --model tiny
python practice/04_vit/05_analyze.py
```

핵심 확인:

```text
16x16 patch
patch embedding
CLS token
position embedding
Multi-Head Attention
MLP
residual connection
attention visualization
```

---

## 05. DINOv2

논문: Learning Robust Visual Features without Supervision

데이터: Oxford-IIIT Pets (논문 evaluation benchmark)

방식: official pretrained feature analysis

먼저 읽을 파일:

```text
practice/05_dinov2/README.md
practice/05_dinov2/common.py
```

실행 순서:

```bash
python scripts/download_torchvision_data.py pets
python practice/05_dinov2/01_data.py
python practice/05_dinov2/02_features.py
python practice/05_dinov2/03_pca.py
python practice/05_dinov2/04_probe.py
python practice/05_dinov2/05_analyze.py
```

핵심 확인:

```text
CLS feature
patch feature
frozen representation
PCA visualization
linear probe
nearest-neighbor retrieval
```

---

## 06. Segment Anything (SAM)

논문: Segment Anything

데이터: official SA-1B subset

방식: official pretrained SAM analysis

먼저 읽을 파일:

```text
practice/06_sam/README.md
practice/06_sam/sam_utils.py
```

실행 순서:

```bash
bash scripts/download_sam_vit_b.sh
python practice/06_sam/01_image.py
python practice/06_sam/02_point_prompt.py
python practice/06_sam/03_box_prompt.py
python practice/06_sam/04_ambiguity.py
python practice/06_sam/05_analyze.py
```

핵심 확인:

```text
image encoder
prompt encoder
mask decoder
point prompt
box prompt
multimask ambiguity
predicted IoU
```

---

## 07. Diffusion Policy

논문: Diffusion Policy

데이터: official Push-T demonstrations

방식: official code 기반 scaled analysis/training

먼저 읽을 파일:

```text
practice/07_diffusion_policy/README.md
practice/07_diffusion_policy/common.py
```

실행 순서:

```bash
bash scripts/setup_diffusion_policy.sh
bash scripts/download_pusht.sh
python practice/07_diffusion_policy/01_data.py
python practice/07_diffusion_policy/02_diffusion.py
python practice/07_diffusion_policy/03_policy.py
python practice/07_diffusion_policy/04_train.py
python practice/07_diffusion_policy/05_rollout.py
python practice/07_diffusion_policy/06_analyze.py
```

핵심 확인:

```text
observation horizon
action horizon
prediction horizon
noise addition
noise prediction
conditional denoising
receding-horizon execution
multimodal action samples
```

---

## 권장 진행 방식

한 논문의 모든 파일을 빠르게 실행하고 넘어가는 것이 목적이 아니다.

각 논문에서 최소한 다음을 설명할 수 있을 때 다음 논문으로 넘어간다.

```text
왜 이 모델이 필요한가?
모델 핵심 아이디어는 무엇인가?
input / GT / output / loss는 무엇인가?
논문 구조가 코드 어디에 구현되어 있는가?
실험 결과는 논문 주장을 어떻게 뒷받침하는가?
내가 직접 본 feature / prediction / failure case는 무엇인가?
```
