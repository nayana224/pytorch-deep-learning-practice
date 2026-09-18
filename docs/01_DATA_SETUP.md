# 01. 데이터 준비

`practice/` 코드는 자동 다운로드하지 않는다.  
dataset/checkpoint 준비는 `scripts/`에서만 수행한다.

## 01 ResNet

```bash
python scripts/download_torchvision_data.py cifar10
```

```text
data/01_resnet/
```

## 02 U-Net

```bash
bash scripts/download_isbi2012.sh
```

```text
data/02_unet/isbi2012/
```

## 03 DeepLabv3+

PASCAL VOC 2012:

```bash
bash scripts/download_voc2012.sh
```

공개 pretrained reality check:

```bash
bash scripts/setup_deeplabv3plus_pretrained.sh
```

## 04 Attention Is All You Need

외부 dataset이 필요 없다.  
작은 deterministic tensor로 Q/K/V와 attention을 확인한다.

## 05 ViT

```bash
python scripts/download_torchvision_data.py cifar100
```

```text
data/05_vit/
```

## 06 SAM

```bash
bash scripts/setup_sam.sh
bash scripts/download_sam_vit_b.sh
```

SA-1B subset은 라이선스 동의 후 직접 준비한다.

```text
data/06_sam/sa1b/
```

## 07 DINOv2

```bash
bash scripts/setup_dinov2.sh
python scripts/download_torchvision_data.py pets
```

```text
external/dinov2/
data/07_dinov2/
```

official repo와 pretrained weight는 setup 단계에서 준비한다.
practice 실행 중에는 GitHub repo를 자동으로 받지 않는다.

## 08 ACT

현재 즉시 실행하는 mechanism visualization은 외부 dataset이 필요 없다.

주의: `01_action_chunking.py`, `02_temporal_ensemble.py`, `03_cvae_latent.py`는 **toy mechanism check**다.  
실제 Level 3 ACT/ALOHA analysis는 official dataset/model 연결 단계에서 별도로 진행한다.

## 09 DDPM

논문 실험에 실제 등장하는 CIFAR-10을 사용한다.

```bash
python scripts/download_torchvision_data.py ddpm
```

```text
data/09_ddpm/
```

## 10 Diffusion Policy

```bash
bash scripts/setup_diffusion_policy.sh
bash scripts/download_pusht.sh
```

```text
data/10_diffusion_policy/
```

## 결과 파일

dataset과 결과 파일을 섞지 않는다.

```text
data/<paper>/      raw input
outputs/<paper>/   png / json / checkpoint
```
