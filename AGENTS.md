# AGENTS.md

## 프로젝트 목적
이 저장소는 PyTorch 딥러닝 학습용 실습 저장소다. 코드 완성 자체보다 논문의 문제의식, 데이터 흐름, 모델 구조, loss, evidence를 직접 실행하며 이해하는 것을 우선한다.

사용자가 코드를 읽고, 실행하고, 핵심 부분을 직접 다시 타이핑하며 이해하는 것이 목표다. AI/Codex는 대리 구현자에 그치지 않고 실습 설계자, 코드 리뷰어, 디버깅 파트너 역할을 우선한다.

## 저장소 역할 분리
- `lessons/`: MLP, CNN, Attention, Transformer 등 일반 딥러닝 기초와 문법 실습
- `practice/`: 논문 단위 실습
- `outputs/`: 실행 결과 이미지/모델/로그
- `scripts/`: 환경 설정, 데이터 획득, 점검 스크립트
- `external/`: 필요할 때 clone되는 공식/외부 소스. 원본 구조 보존

## practice 공통 원칙
- 모든 `practice/` 실습은 **반드시 해당 논문에 실제로 등장하는 dataset / benchmark / task를 사용**한다.
- 다른 임의 dataset으로 편의상 대체하지 않는다.
- 논문이 실제 사용한 input/GT, architecture, loss, augmentation, metric을 가능한 한 기준으로 삼는다.
- raw data를 보기 전에 synthetic data로 대체하지 않는다. synthetic data는 shape/debugging 진단용으로만 허용한다.
- 논문 전체 재현이 비현실적이면 `Faithful / Scaled / Pretrained analysis` 중 어느 수준인지 README에 명시한다.
- large-scale pretraining dataset(JFT-300M, LVD-142M, SA-1B 전체 등)을 로컬에서 재학습할 수 없으면, 논문 공식 pretrained model을 사용하고 논문에 실제 등장하는 downstream/evaluation dataset으로 분석한다.
- 논문과 다른 선택을 했으면 이유와 차이를 README에 기록한다.
- 비교 실험에서는 한 번에 한 조건만 바꾼다.
- accuracy 한 숫자보다 tensor shape, feature, mask, attention, prediction, error case를 적극적으로 관찰한다.

## 코드 읽기 + 재타이핑 실습 방식
`practice/`는 TODO 빈칸 채우기 방식이 아니다.

기본 사이클:
1. AI가 해당 논문 실습에 필요한 실행 가능한 전체 코드/파일 세트를 구성할 수 있다.
2. 사용자는 먼저 코드를 읽으며 data flow와 핵심 연산을 분석한다.
3. 시각화/print/checkpoint 지점을 코드 안에 명확히 둔다.
4. 사용자가 직접 실행하고 shape, feature, prediction, failure case를 관찰한다.
5. 이해되지 않는 줄이나 메커니즘을 질문한다.
6. AI가 코드의 이유, 논문과의 대응, failure mode를 설명한다.
7. 충분히 이해한 뒤 사용자가 핵심 코드를 스스로 다시 타이핑해보며 복습한다.

## 현재 practice 구조와 paper-data 기준
- `01_resnet`: CIFAR-10 paper experiment. Plain vs residual, 6n+2 architecture, option-A shortcut, paper optimizer/schedule.
- `02_unet`: ISBI 2012 EM segmentation challenge. Original valid-conv U-Net, crop+concat, TIFF stack.
- `03_deeplabv3plus`: PASCAL VOC 2012 우선, Cityscapes 확장. ASPP + low-level decoder + atrous separable conv.
- `04_vit`: 논문 downstream 중 CIFAR-100 우선. ViT-B/16 구조와 patch/token/attention 분석. JFT/ImageNet-21k pretraining은 scaled 또는 pretrained analysis로 구분.
- `05_dinov2`: official pretrained DINOv2 + 논문 benchmark인 Oxford-IIIT Pets를 기본 분석 dataset으로 사용. PCA patch features, frozen linear probe, retrieval.
- `06_sam`: official pretrained SAM + SA-1B image/mask subset. point/box/multimask ambiguity와 IoU를 분석. SA-1B 전체 재학습은 하지 않는다.
- `07_diffusion_policy`: official Push-T demonstration dataset. observation/action horizons, DDPM action denoising, receding-horizon rollout을 분석한다.

## 데이터 확인 규칙
가능하면 항상 다음 순서로 본다.
`raw data → file structure → transform → tensor → batch → model input → output → GT → loss`

각 단계에서 확인:
- shape
- dtype
- value range / unit
- channel 의미
- label/mask/action 의미
- invalid/missing 값
- train/val/test split 또는 episode split

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
- attention / PCA / probability
- error map / failure case
- 논문 핵심 메커니즘에 해당하는 시각화

## 공개/공식 코드
- 공식 구현이 있으면 우선 확인하고, foundation model 계열(DINOv2, SAM, Diffusion Policy)은 공식 checkpoint/code를 적극 활용한다.
- 공식 코드가 현재 학습 프레임워크와 다르면 원본을 `external/`에 보존하고 핵심 구조를 PyTorch로 재구현할 수 있다.
- 공식 notebook은 단순 형식 변환 목적으로 임의 수정하지 않는다.

## 새 작업 규칙
새로운 작업을 할 때마다 본 파일과 해당 README가 실제 저장소 상태 및 학습 방식과 맞는지 확인하고 필요한 경우 함께 업데이트한다.
