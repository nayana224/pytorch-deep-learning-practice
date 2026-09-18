# 03. DeepLabv3+ — Encoder-Decoder with Atrous Separable Convolution

## Paper claim

이 실습에서 확인할 논문 주장은 두 가지다.

1. **Atrous Spatial Pyramid Pooling (ASPP)**은 서로 다른 dilation rate와 image-level feature를 함께 사용해 multi-scale context를 만든다.
2. **DeepLabv3+ decoder**는 high-level semantic feature에 low-level feature를 결합해 object boundary를 더 정교하게 복원한다.

단순히 segmentation이 되는지만 보지 않고, 위 두 주장이 실제 feature와 prediction에서 관찰되는지를 확인한다.

## Target configuration

실습을 두 층으로 나눈다.

### A. 구조/메커니즘 확인용

기준 모델은 **output stride 16의 scaled DeepLabv3+ decoder 구조**다.

- `v3plus`: ASPP + low-level 48-channel projection + decoder
- `v3`: ASPP 뒤에서 바로 분류하는 no-decoder baseline

두 모델 비교는 논문의 전체 ablation을 그대로 재현한 것이 아니라 **decoder 효과를 확인하기 위한 controlled scaled comparison**이다.

### B. 실제 prediction 품질 확인용

강한 prediction을 보기 위해 공개 pretrained model인 **DeepLabV3Plus-ResNet101, VOC2012 Aug, OS=16**을 별도 분석한다.

사용 출처:
- VainF/DeepLabV3Plus-Pytorch
- 공개 checkpoint reported mIoU: 0.783 on Pascal VOC2012 Aug validation setup

이 모델은 DeepLabv3+ 논문의 공식 저자 checkpoint가 아니고, 논문의 최종 Xception-65 최고 설정과도 동일하지 않다. 대신 현재 PyTorch 환경에서 쉽게 재현 가능한 공개 pretrained DeepLabv3+ 중 강한 설정을 사용해 **"제대로 학습된 DeepLabv3+가 실제로 어떤 segmentation을 내는가"**를 관찰하는 용도다.

논문의 최종 최고 성능 자체를 재현했다고 해석하지 않는다.

## Reproduction level

**Scaled paper implementation**.

논문 데이터는 PASCAL VOC 2012와 Cityscapes다. 기본 코드는 실제 **PASCAL VOC 2012** image/semantic mask를 사용한다.

full modified Xception-65와 대규모 pretraining/extra annotations는 그대로 재현하지 않는다. 대신 다음 paper mechanism은 유지한다.

- Xception-style separable convolution
- output stride
- ASPP
- low-level feature 48-channel reduction
- high-level + low-level concat
- two 3×3 256-channel decoder convolutions
- pixel-wise cross entropy
- mIoU

## 데이터 준비

공부 코드는 데이터를 자동 다운로드하지 않는다.

```bash
bash scripts/download_voc2012.sh
```

이미 `VOCdevkit/VOC2012`가 준비되어 있으면 다운로드를 건너뛴다.

## What to observe

### 1. Mechanism visualization

`02_atrous.py`

- dilation이 커질 때 kernel sampling 간격이 어떻게 벌어지는가?
- parameter 수를 늘리지 않고 더 넓은 spatial context를 보는 이유가 무엇인가?

`05_analyze.py`

- ASPP의 `1×1 / rate 6 / rate 12 / rate 18 / image pooling` branch가 서로 다른 feature response를 만드는가?
- backbone low-level feature와 ASPP high-level feature의 spatial resolution 차이는 무엇인가?
- low-level feature가 48 channels로 줄어든 뒤 어디에서 concat되는가?

### 2. Prediction visualization

`05_analyze.py`

한 validation sample에 대해 다음을 한 화면에서 본다.

- Input
- GT
- Prediction
- Max class probability
- 전체 error map
- GT boundary에서의 error map

### 3. Evidence visualization

`06_evidence.py`

동일한 scaled backbone/ASPP에서 다음 두 모델을 비교한다.

- DeepLabv3-like no-decoder baseline
- DeepLabv3+ decoder

비교 metric:

- mIoU
- GT boundary pixel accuracy
- interior pixel accuracy

관찰 질문:

> decoder를 넣었을 때 전체 mIoU뿐 아니라 특히 boundary 쪽 정확도가 어떻게 변하는가?

이 결과가 논문의 전체 ablation table을 대체하지는 않는다. 다만 **low-level decoder가 boundary refinement에 기여한다는 핵심 주장을 로컬 실험에서 점검하는 근거**로 사용한다.

## 파일

- `deeplabv3plus.py`: scaled Xception backbone + ASPP + optional decoder
- `01_data.py`: VOC2012 image/GT 및 crop/scale/flip
- `02_atrous.py`: dilation sampling/receptive-field 시각화
- `03_model.py`: feature shape trace
- `04_train.py`: `v3plus` 또는 `v3` 학습, CE/mIoU, training curve 저장
- `05_analyze.py`: ASPP branch / feature flow / prediction / boundary error 시각화
- `06_evidence.py`: decoder 유무 controlled comparison
- `07_pretrained_analyze.py`: 공개 pretrained DeepLabV3Plus-ResNet101의 실제 prediction / confidence / error / boundary error

## 실행

먼저 기본 DeepLabv3+를 학습한다.

```bash
python practice/03_deeplabv3plus/01_data.py
python practice/03_deeplabv3plus/02_atrous.py
python practice/03_deeplabv3plus/03_model.py

python practice/03_deeplabv3plus/04_train.py \
  --variant v3plus \
  --epochs 20

python practice/03_deeplabv3plus/05_analyze.py
```

scaled decoder 효과까지 확인하려면 baseline도 같은 epoch로 학습한다.

```bash
python practice/03_deeplabv3plus/04_train.py \
  --variant v3 \
  --epochs 20

python practice/03_deeplabv3plus/06_evidence.py \
  --max-samples 100
```

실제 잘 학습된 DeepLabv3+ prediction을 보려면 공개 pretrained 모델을 준비하고 분석한다.

```bash
bash scripts/setup_deeplabv3plus_pretrained.sh
python practice/03_deeplabv3plus/07_pretrained_analyze.py
```

## Outputs

`outputs/03_deeplabv3plus/`에 다음 결과가 생긴다.

```text
v3plus.pt
v3.pt

04_history_v3plus.json
04_history_v3.json
04_training_curves_v3plus.png
04_training_curves_v3.png

05_prediction_analysis.png
05_aspp_branches.png
05_feature_flow.png

06_decoder_evidence.json
06_decoder_evidence.png

07_pretrained_prediction_analysis.png
```

이 이미지들이 논문 노트의 다음 항목에 직접 들어갈 실습 근거다.

- 내가 직접 한 실습
- Feature / Prediction Visualization
- 틀린 사례와 원인 추정
- My observation

## Paper vs practice

### 논문과 같은 점

- PASCAL VOC 2012 사용
- atrous convolution
- ASPP의 multi-rate 구조
- output stride 16
- low-level feature → 48 channels
- decoder concat + refinement
- semantic segmentation CE loss
- mIoU 평가

### 축소/변경된 점

- full modified Xception-65가 아니라 scaled Xception-style backbone
- ImageNet/JFT 계열 대규모 pretraining 미사용
- VOC trainaug/SBD extra annotations 미사용
- 논문의 전체 training schedule 미재현
- multi-scale + flip inference 미사용
- decoder evidence는 논문 표 자체의 수치 재현이 아니라 로컬 controlled comparison
- pretrained prediction track은 제3자 공개 PyTorch 재현 구현의 ResNet101 checkpoint이며, 논문 저자의 최종 Xception-65 checkpoint와 동일하지 않음

따라서 이 실습으로 주장할 수 있는 것은 **논문 구조와 핵심 메커니즘을 이해하고, scaled 조건에서 그 효과를 직접 관찰했다**는 수준이다. 논문의 최종 benchmark 수치 재현으로 해석하면 안 된다.

다음 fidelity 단계는 VOC trainaug(SBD), exact Xception-65 middle-flow depth, Cityscapes fine/coarse training, multi-scale+flip inference다.
