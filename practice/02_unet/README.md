# 02. U-Net — Convolutional Networks for Biomedical Image Segmentation

이 폴더는 U-Net 논문의 실제 실험 흐름을 PyTorch로 따라간다.

## 재현 수준
기본 목표는 **Faithful reproduction에 가까운 학습 실습**이다.

가능하면 논문과 동일하게 다음을 따른다.
- dataset: ISBI 2012 EM segmentation challenge
- architecture: original U-Net의 valid convolution / crop + concat 구조
- task: pixel-wise binary segmentation
- augmentation: elastic deformation
- loss: pixel-wise cross entropy + boundary-aware weight map
- evaluation: challenge 지표를 우선 이해하고, IoU/Dice는 보조 지표로 사용

단, 원 논문의 Caffe training 환경과 전체 실험 서버를 그대로 복원하는 것이 목적은 아니다. PyTorch에서 데이터 흐름과 핵심 메커니즘을 직접 재구성한다.

## 1. Dataset

### 공식 source
ISBI 2012 challenge의 원래 서버는 종료되었지만, ImageJ에서 training/test data archive를 계속 제공한다.

- challenge 설명: `https://imagej.net/events/isbi-2012-segmentation-challenge`
- dataset archive: `https://downloads.imagej.net/ISBI-2012-challenge.zip`

논문/챌린지 기준 training data는 다음과 같다.
- 30개의 serial-section TEM slice
- 각 slice 크기: 512 x 512
- Drosophila first instar larva ventral nerve cord
- 각 training slice에 binary segmentation label 존재
- label은 white=segmented object 내부, black=주로 membrane/background
- test volume은 같은 specimen의 다른 volume이며 공개 GT는 제공되지 않음

### 저장 위치
repo root에서 아래 위치를 사용한다.

```text
data/
└── 02_unet/
    └── isbi2012/
        └── <archive에서 풀린 TIFF files>
```

`data/`는 `.gitignore`에 포함되어 있으므로 원본 dataset을 GitHub에 commit하지 않는다.

### 다운로드
repo root에서 실행:

```bash
bash scripts/download_isbi2012.sh
```

직접 받고 싶으면:

```bash
mkdir -p data/02_unet/isbi2012
cd data/02_unet/isbi2012
wget https://downloads.imagej.net/ISBI-2012-challenge.zip
unzip ISBI-2012-challenge.zip
```

다운로드 후에는 먼저 파일명을 확인한다.

```bash
find data/02_unet/isbi2012 -maxdepth 2 -type f | sort
```

archive의 실제 파일명을 코드에서 추측하지 않는다. 먼저 `find` 결과를 보고 `01_data.py`의 경로를 정한다.

## 2. 첫 실습: `01_data.py`
아직 모델을 만들지 않는다.

첫 목표는 raw dataset 자체를 이해하는 것이다.

확인 순서:
1. training image TIFF stack 열기
2. GT TIFF stack 열기
3. stack shape 확인
4. slice 하나의 shape / dtype / min / max 확인
5. GT unique value 확인
6. image와 같은 index의 GT를 나란히 표시
7. foreground/background pixel 비율 확인

반드시 이해할 것:
- `[30, 512, 512]`에서 각 축이 무엇을 의미하는가?
- grayscale image와 binary GT의 dtype/range가 왜 다른가?
- GT의 white/black pixel이 의미하는 class는 무엇인가?
- classification label과 segmentation GT가 어떻게 다른가?

## 3. Network
`02_model.py`에서는 논문 Figure 1의 original U-Net을 직접 구현한다.

핵심 흐름:

```text
input
→ valid 3x3 conv + ReLU
→ valid 3x3 conv + ReLU
→ 2x2 max pool
→ contracting path
→ bottleneck
→ up-convolution
→ encoder feature crop
→ channel-wise concat
→ expansive path
→ 1x1 conv
→ pixel logits
```

주의: dataset slice `512x512`와 Figure 1의 network input tile `572x572 → 388x388 output`은 같은 개념이 아니다. dataset을 읽은 뒤 tile/context handling을 별도로 확인한다.

## 4. Training / Analysis

`03_train.py`
- raw image / GT → tensor
- model output과 GT spatial alignment
- pixel-wise loss
- backward / optimizer
- 작은 subset overfit sanity check
- 이후 논문의 weighted loss / elastic deformation 추가

`04_analyze.py`
- input / GT / prediction
- probability map
- error map
- encoder/decoder feature
- failure case
- challenge metric 개념과 IoU/Dice 비교

## 완료 기준
1. Problem: sliding-window CNN의 중복 계산과 localization/context trade-off
2. Core idea: contracting path + symmetric expanding path + skip feature
3. Method: valid conv / pool / up-conv / crop / concat / 1x1 conv
4. Input / GT / Output / Loss: EM image → logits, segmentation map → pixel loss
5. Evidence: 논문 지표/결과와 직접 prediction 관찰
6. My observation: feature와 failure case에서 직접 본 현상
