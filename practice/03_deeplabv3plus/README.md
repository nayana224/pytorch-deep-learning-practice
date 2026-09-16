# 03. DeepLabv3+ — Encoder-Decoder with Atrous Separable Convolution

재현 수준: **Scaled paper implementation**.

논문 데이터는 PASCAL VOC 2012와 Cityscapes다. 기본 코드는 실제 **PASCAL VOC 2012** image/semantic mask를 사용한다. 논문의 full modified Xception-65 및 대규모 pretraining/extra annotations를 그대로 재현하는 비용은 크므로 backbone의 middle-flow 반복 수를 줄였지만, paper mechanism인 Xception-style separable conv, output stride, ASPP, low-level 48-channel reduction, concat, 두 개의 3×3 256 decoder conv는 유지한다.

논문 decoder 설계의 핵심은 OS=16의 DeepLabv3 feature를 low-level feature 크기로 upsample하고, low-level channel을 1×1 conv로 48로 줄인 뒤 concat하여 두 개의 3×3 256 conv로 refinement하는 것이다.

## 파일
- `deeplabv3plus.py`: scaled Xception + ASPP + paper decoder
- `01_data.py`: VOC2012 image/GT 및 crop/scale/flip
- `02_atrous.py`: dilation에 따른 sampling/receptive-field 시각화
- `03_model.py`: feature shape trace
- `04_train.py`: pixel CE(ignore=255), mIoU
- `05_analyze.py`: input/GT/pred/error + intermediate feature shape

## 실행
```bash
python practice/03_deeplabv3plus/01_data.py
python practice/03_deeplabv3plus/02_atrous.py
python practice/03_deeplabv3plus/03_model.py
python practice/03_deeplabv3plus/04_train.py --epochs 20
python practice/03_deeplabv3plus/05_analyze.py
```

다음 fidelity 단계는 VOC trainaug(SBD), exact Xception-65 middle-flow depth, Cityscapes fine/coarse training, multi-scale+flip inference다.
