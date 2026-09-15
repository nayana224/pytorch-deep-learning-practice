# AGENTS.md

## 프로젝트 목적
이 저장소는 PyTorch 딥러닝 학습용 실습 저장소다. 코드 완성 자체보다 논문의 문제의식, 데이터 흐름, 모델 구조, loss, evidence를 직접 실행하며 이해하는 것을 우선한다.

사용자가 직접 코드를 타이핑하며 공부하는 것이 핵심 목표다. AI/Codex는 대리 구현자가 아니라 실습 설계자, 코드 리뷰어, 디버깅 파트너 역할을 우선한다.

## 저장소 역할 분리
- `lessons/`: MLP, CNN, Attention, Transformer 등 일반 딥러닝 기초와 문법 실습
- `practice/`: 논문 단위 실습
- `outputs/`: 실행 결과 이미지/모델/로그
- `scripts/`: 환경 설정, 데이터 획득, 점검 스크립트
- `external/`: 필요할 때 clone되는 공식/외부 소스. 원본 구조 보존

## practice 공통 원칙
- 논문이 실제 사용한 dataset, input/GT, architecture, loss, augmentation, metric을 가능한 한 기준으로 삼는다.
- raw data를 보기 전에 임의 synthetic data로 대체하지 않는다.
- synthetic data는 pipeline/debugging 분리를 위한 최소 진단용으로만 사용한다.
- 논문 전체 재현이 비현실적이면 `Faithful / Scaled / Pretrained analysis` 중 어느 수준인지 README에 명시한다.
- 논문과 다른 선택을 했으면 이유와 차이를 기록한다.
- 대규모 foundation model은 공식 pretrained model 분석을 우선하되, training objective와 데이터 흐름은 논문 기준으로 설명한다.
- 비교 실험에서는 한 번에 한 조건만 바꾼다.
- accuracy 한 숫자보다 tensor shape, feature, mask, attention, prediction, error case를 적극적으로 관찰한다.

## 직접 타이핑 실습 방식
`practice/`는 TODO 빈칸 채우기 방식이 아니다.

다음 사이클을 반복한다.
1. AI가 현재 단계에 필요한 실제 코드를 대화에서 제시한다.
2. 사용자가 그 코드를 직접 `.py` 파일에 타이핑한다.
3. 실행 전에 가능한 경우 shape, 출력 의미, 변화 방향을 예상한다.
4. 사용자가 직접 실행한다.
5. 실행 결과와 이해되지 않는 줄을 질문한다.
6. AI가 결과를 해석하고 코드의 이유, failure mode, 다음 관찰 지점을 설명한다.
7. 이해가 끝나면 다음 단계로 넘어간다.

원칙:
- 전체 프로젝트의 완성 코드를 처음부터 한 번에 던지지 않는다.
- API 이름을 맞히는 퀴즈처럼 TODO만 남기지도 않는다.
- 파일 중간에 어디에 붙여야 할지 헷갈릴 수 있으므로, 한 단계가 진행될 때는 해당 `.py` 파일의 **현재 전체 코드**를 통째로 제시하는 것을 기본으로 한다.
- 단, 전체 파일을 주더라도 한 번에 새로 추가되는 개념은 가능한 한 하나의 작은 학습 단위로 제한한다.
- 사용자는 전체 파일을 직접 타이핑하거나 현재 파일과 비교해 갱신하고, 이해되지 않는 부분을 질문한다.
- 최종 파일에는 사용자가 단계적으로 직접 작성한 코드가 남도록 한다.
- 이미 충분히 이해한 boilerplate는 반복 설명하지 않는다.

## 현재 practice 구조
```text
practice/
├── 01_resnet/
├── 02_unet/
├── 03_deeplabv3plus/
├── 04_vit/
├── 05_dinov2/
├── 06_sam/
└── 07_diffusion_policy/
```

### 01_resnet
- 논문: Deep Residual Learning for Image Recognition
- 우선 데이터: CIFAR-10 (논문 실험에 포함)
- 핵심 검증: plain vs residual, degradation/optimization
- 기존에 사용자가 직접 타이핑한 `residual_block.py`는 학습 이력으로 보존

### 02_unet
- 논문: U-Net: Convolutional Networks for Biomedical Image Segmentation
- 우선 데이터: ISBI 2012 EM segmentation challenge
- 공식 설명 페이지: `https://imagej.net/events/isbi-2012-segmentation-challenge`
- 현재 archive: `https://downloads.imagej.net/ISBI-2012-challenge.zip`
- dataset helper: `scripts/download_isbi2012.sh`
- local data 위치: `data/02_unet/isbi2012/` (`data/`는 gitignore)
- training: 30장의 512x512 EM image + fully annotated segmentation map
- dataset을 받은 뒤 실제 archive 파일명을 먼저 `find`로 확인하고 코드 경로를 정한다. 파일명을 미리 가정하지 않는다.
- 현재 확인된 파일: `train-volume.tif`, `train-labels.tif`, `test-volume.tif`, `test-labels.tif`, `challenge-error-metrics.bsh`
- TIFF는 PNG로 변환하지 않고 multi-page stack 그대로 먼저 읽는다.
- `01_data.py` 첫 단계에서는 train image/label stack의 shape, dtype, min/max, label unique value를 확인하고, 같은 index의 image/GT를 `matplotlib`로 나란히 시각화한다.
- label 값이 `[0, 255]`처럼 binary로 보이더라도 어느 값이 membrane/cell interior인지 미리 단정하지 않고 image와 GT를 확대/overlay하여 시각적으로 먼저 검증한다.
- 현재 관찰상 `0`은 membrane/boundary, `255`는 cell interior로 해석한다.
- 다음 단계에서는 raw `uint8` image/label을 PyTorch 학습 입력으로 바꾸는 과정을 직접 확인한다: image는 `float32` 및 0~1 스케일로 변환하고 channel 축을 추가하며, GT는 논문 class 의미를 유지하면서 학습용 class/tensor 형태로 변환한다.
- 변환 전후에 반드시 shape, dtype, value range, unique value를 출력해 확인한다.
- Figure 1 input tile 572x572와 dataset image 512x512를 구분
- 구조: valid conv / pool / up-conv / crop + concat / 1x1 conv
- 이후 weighted loss, elastic deformation, overlap-tile을 논문 순서로 추가

### 03_deeplabv3plus
- 논문: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
- 우선 데이터: PASCAL VOC 2012, 이후 Cityscapes
- 핵심 검증: atrous convolution, ASPP, low-level decoder feature, output stride, mIoU

### 04_vit
- 논문: An Image Is Worth 16x16 Words
- 논문 데이터: ImageNet/ImageNet-21k/JFT-300M 및 downstream datasets
- 로컬 실습: 논문에 포함된 공개 downstream dataset으로 patch/token 흐름과 transfer를 검증
- 대규모 pretraining을 생략하면 Scaled 또는 Pretrained analysis로 명시

### 05_dinov2
- 논문: DINOv2: Learning Robust Visual Features without Supervision
- 원 논문 pretraining: LVD-142M + large ViT SSL
- 실습: 공식 pretrained DINOv2 feature 분석 중심
- PCA patch visualization, frozen feature, nearest-neighbor/linear probe 등을 우선

### 06_sam
- 논문: Segment Anything
- 원 논문: SA-1B 11M images / 1.1B masks, data engine
- 실습: 공식 pretrained SAM으로 image encoder / prompt encoder / mask decoder와 promptable segmentation 분석
- point / box / ambiguity / multiple masks / failure case를 관찰

### 07_diffusion_policy
- 논문: Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
- 우선: 공개 demonstration/benchmark 중 재현 가능한 task
- 핵심 흐름: observation + noisy action + diffusion timestep → predicted noise → MSE
- action sequence, receding horizon, multimodality, rollout failure를 관찰

## 공개/공식 코드
- 외부 공식 구현이 있으면 먼저 언어, 프레임워크, 파일 구조, 논문과의 대응을 확인한다.
- 공식 코드가 현재 학습 프레임워크와 다르면 원본을 보존하고 핵심 구조를 PyTorch로 다시 구현할 수 있다.
- 공식 notebook은 형식만 바꾸기 위해 임의로 `.py`로 변환하지 않는다.
- 외부 소스는 `external/`에서 원본을 임의 수정하지 않는다.

## 데이터 확인 규칙
가능하면 항상 다음 순서로 본다.
`raw data → file structure → transform → tensor → batch → model input → output → GT → loss`

각 단계에서 확인:
- shape
- dtype
- value range / unit
- channel 의미
- label/mask 의미
- invalid/missing 값
- train/val/test split

## 논문 실습 완료 기준
다음 6가지를 자신의 말로 설명하고 최소 실습 1개를 완료해야 한다.
1. Problem
2. Core idea
3. Method
4. Input / GT / Output / Loss
5. Evidence
6. My observation

## 시각화 우선 원칙
- input / GT
- tensor shape 흐름
- intermediate feature
- prediction
- metric curve
- error map / failure case
- 논문 핵심 메커니즘에 해당하는 시각화

## 새 작업 규칙
새로운 작업을 할 때마다 본 파일과 해당 README가 실제 저장소 상태 및 학습 방식과 맞는지 확인하고 필요한 경우 함께 업데이트한다.
