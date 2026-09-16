# 02. U-Net — Convolutional Networks for Biomedical Image Segmentation

이 폴더는 U-Net 논문의 핵심 구조와 ISBI 2012 EM 데이터 흐름을 PyTorch로 한 번에 실행하고 분석하기 위한 실습이다.

## 현재 실습 전략

이번 단계부터는 코드를 작은 조각으로 이어 붙이기보다 **실행 가능한 전체 파이프라인을 먼저 제공하고, 사용자가 코드를 읽고 시각화/shape/output을 분석한 뒤 핵심 부분을 다시 타이핑하는 방식**으로 진행한다.

논문에서 직접 확인할 핵심은 다음과 같다.

- contracting path: valid 3x3 conv + ReLU, 2x2 max pool
- channel: 64 → 128 → 256 → 512 → 1024
- expanding path: up-convolution → encoder feature crop → channel concat → double conv
- final 1x1 convolution
- pixel-wise softmax + cross entropy
- 매우 적은 biomedical training data
- elastic deformation augmentation
- touching cell separation을 강조하는 boundary-aware weighted loss

논문은 contracting path와 symmetric expanding path를 사용하고, valid convolution 때문에 encoder feature를 crop한 뒤 decoder feature와 concatenate한다. 최종 1x1 convolution은 64-component feature를 class logits로 바꾼다. 전체는 23개의 convolutional layer로 구성된다.

## Dataset

- ISBI 2012 EM segmentation challenge
- train volume: 30 slices
- slice size: 512 x 512
- raw image: grayscale EM
- GT: white=cell interior, black=membrane

다운로드:

```bash
bash scripts/download_isbi2012.sh
```

현재 확인된 archive 파일:

```text
data/02_unet/isbi2012/
├── train-volume.tif
├── train-labels.tif
├── test-volume.tif
├── test-labels.tif
└── challenge-error-metrics.bsh
```

TIFF는 PNG로 변환하지 않고 multi-page stack 그대로 사용한다.

## 파일 구조

```text
practice/02_unet/
├── 01_data.py
├── 02_model.py
├── 03_train.py
├── 04_analyze.py
├── unet.py
└── README.md
```

### `01_data.py`

raw TIFF stack을 읽고 다음을 확인한다.

- stack shape / dtype / min / max
- GT unique values
- membrane / cell-interior pixel ratio
- NumPy → PyTorch tensor 변환
- input / GT / membrane overlay 시각화

실행:

```bash
python practice/02_unet/01_data.py
```

### `unet.py`

원 논문 형태에 가까운 U-Net model definition이다.

- valid convolution (`padding=0`)
- four encoder stages
- bottleneck
- four transposed-convolution decoder stages
- center crop
- channel concat
- final 1x1 classifier
- `return_features=True`로 intermediate feature 관찰 가능

### `02_model.py`

논문 Figure 1의 `572x572 → 388x388` shape 흐름을 검증한다.

실행:

```bash
python practice/02_unet/02_model.py
```

특히 확인할 feature:

- `enc1`, `enc4`
- `bottleneck`
- `up4`
- `crop4`
- `concat4`
- `dec1`

생성되는 시각화:

```text
outputs/02_unet/02_model_feature_overview.png
outputs/02_unet/02_model_crop_concat.png
```

## Training baseline

`03_train.py`는 실제 30장의 EM image/GT를 model과 연결한다.

```bash
python practice/02_unet/03_train.py --epochs 10
```

현재 baseline 설정:

- first 24 slices: train
- last 6 slices: validation
- batch size: 1
- optimizer: SGD
- momentum: 0.99
- loss: standard `CrossEntropyLoss`
- augmentation: flip + 90-degree rotation
- metric: membrane IoU

중요: 이것은 **원 논문 training의 완전 재현이 아니다.**

논문은 pixel-wise cross entropy에 class-frequency weighting과 touching-cell separation border를 강조하는 per-pixel weight map을 사용했고, elastic deformation을 강하게 사용했다. 현재 baseline은 architecture/data-flow를 먼저 분석하기 위해 boundary-aware weight map과 elastic deformation을 아직 넣지 않았다.

또한 dataset이 serial-section volume이므로 24/6 slice split은 독립적인 benchmark split이라고 주장할 수 없다. 학습 파이프라인 검증용이다.

checkpoint:

```text
outputs/02_unet/unet_baseline.pt
```

training curve:

```text
outputs/02_unet/03_train_curves.png
```

## Prediction / feature analysis

training 후:

```bash
python practice/02_unet/04_analyze.py
```

확인 항목:

- center-cropped input field-of-view
- aligned GT
- membrane probability map
- prediction
- prediction overlay
- error map
- membrane IoU / Dice
- encoder / bottleneck / skip / decoder feature

생성되는 파일:

```text
outputs/02_unet/04_prediction_analysis.png
outputs/02_unet/04_feature_analysis.png
```

## 왜 GT를 crop하는가?

original U-Net은 valid convolution을 사용하므로 model output이 input보다 작다.

Figure 1 예시:

```text
input  : 572 x 572
output : 388 x 388
```

따라서 loss 계산 시 full-size GT에서 output과 같은 central field-of-view를 crop해 alignment해야 한다.

현재 512x512 dataset slice를 그대로 model input으로 넣으면 output spatial size 역시 더 작아지며, `center_crop_target()`이 GT를 output 크기에 맞춘다.

## 완료 기준

1. Problem: sliding-window CNN의 redundant computation과 localization/context trade-off
2. Core idea: contracting path + symmetric expanding path + high-resolution skip feature
3. Method: valid conv / pool / up-conv / crop / concat / 1x1 conv
4. Input / GT / Output / Loss: EM image → pixel logits, segmentation map → aligned pixel-wise CE
5. Evidence: 논문 결과와 직접 얻은 probability/prediction/feature 비교
6. My observation: crop+concat, feature 변화, membrane failure case를 직접 설명

## 다음 fidelity 단계

전체 baseline을 이해한 뒤 논문 재현도를 높이는 순서는 다음과 같다.

1. elastic deformation
2. boundary-aware per-pixel weight map
3. overlap-tile inference + mirror padding
4. challenge metric과 논문 결과 비교
