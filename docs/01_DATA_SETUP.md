# 01. 데이터 준비

논문 실습을 실행하기 전에 필요한 데이터와 checkpoint를 먼저 준비한다.

## 원칙

- `practice/*.py`에서는 자동 다운로드하지 않는다.
- 다운로드는 `scripts/`에서만 한다.
- 데이터가 이미 완전히 준비되어 있으면 `[skip]` 하고 종료한다.
- 다운로드 중 끊겼을 때 가능한 경우 이어받기(resume)를 사용한다.

## 1. ResNet — CIFAR-10

```bash
python scripts/download_torchvision_data.py cifar10
```

준비 위치:

```text
data/01_resnet/
```

이미 정상 데이터가 있으면 다시 받지 않는다.

## 2. U-Net — ISBI 2012 EM

```bash
bash scripts/download_isbi2012.sh
```

준비 위치:

```text
data/02_unet/isbi2012/
```

핵심 파일:

```text
train-volume.tif
train-labels.tif
test-volume.tif
test-labels.tif
```

이미 추출된 핵심 파일이 있으면 다시 다운로드/압축해제하지 않는다.

## 3. DeepLabv3+ — PASCAL VOC 2012

```bash
bash scripts/download_voc2012.sh
```

준비 위치:

```text
data/03_deeplabv3plus/VOCdevkit/VOC2012/
```

이 데이터는 크므로 다운로드가 오래 걸릴 수 있다. `wget -c` 또는 `curl -C -` 방식으로 이어받기를 사용한다.

완전히 준비되면 다음 파일이 존재해야 한다.

```text
data/03_deeplabv3plus/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt
```

## 4. ViT — CIFAR-100

```bash
python scripts/download_torchvision_data.py cifar100
```

준비 위치:

```text
data/04_vit/
```

## 5. DINOv2 — Oxford-IIIT Pets

```bash
python scripts/download_torchvision_data.py pets
```

준비 위치:

```text
data/05_dinov2/
```

DINOv2 model weight는 `torch.hub`가 official model을 처음 사용할 때 별도로 cache한다.

## 6. SAM — SA-1B subset + SAM checkpoint

SAM checkpoint:

```bash
bash scripts/download_sam_vit_b.sh
```

checkpoint 위치:

```text
checkpoints/sam/sam_vit_b_01ec64.pth
```

이미 checkpoint가 있으면 다시 다운로드하지 않는다.

SA-1B는 라이선스/접근 절차가 있으므로 자동 다운로드하지 않는다. 사용자가 받은 official SA-1B shard/subset을 다음 위치에 둔다.

```text
data/06_sam/sa1b/
```

## 7. Diffusion Policy — Push-T demonstrations

먼저 official repository를 준비한다.

```bash
bash scripts/setup_diffusion_policy.sh
```

그 다음 Push-T demonstration을 준비한다.

```bash
bash scripts/download_pusht.sh
```

준비 위치:

```text
data/07_diffusion_policy/
```

`.zarr` dataset이 이미 있으면 다시 받지 않는다.

## 한 번에 확인할 때

필요한 논문만 순서대로 준비하면 된다. 모든 대형 데이터를 한꺼번에 받을 필요는 없다.

예를 들어 현재 DeepLabv3+를 볼 차례라면:

```bash
git pull
bash scripts/download_voc2012.sh
python practice/03_deeplabv3plus/01_data.py
```
