# PyTorch Deep Learning Practice

논문을 **코드로 읽고, 실행하고, 시각화하며 이해하기 위한 PyTorch 실습 저장소**다.

목표는 코드를 복잡하게 잘 짜는 것이 아니라, 논문의 Figure와 Method가 코드에서 바로 보이게 만드는 것이다.

## 시작 순서

처음에는 `docs/`부터 본다.

```text
docs/
├── README.md
├── 01_DATA_SETUP.md
├── 02_STUDY_ORDER.md
└── 03_PAPER_CHECKLIST.md
```

권장 순서:

```text
1. docs/01_DATA_SETUP.md
2. docs/02_STUDY_ORDER.md
3. 각 practice/<paper>/README.md
4. model definition
5. data / model / train / analyze 실행
6. docs/03_PAPER_CHECKLIST.md로 마무리
```

## 현재 논문 실습

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

| 순서 | 논문 | 기본 데이터 | 핵심 |
|---:|---|---|---|
| 1 | ResNet | CIFAR-10 | plain vs residual, F(x)+x, degradation |
| 2 | U-Net | ISBI 2012 EM | valid conv, crop+concat, segmentation |
| 3 | DeepLabv3+ | PASCAL VOC 2012 | atrous conv, ASPP, decoder |
| 4 | ViT | CIFAR-100 | patch, CLS token, Transformer, attention |
| 5 | DINOv2 | Oxford-IIIT Pets | frozen features, PCA, linear probe |
| 6 | SAM | SA-1B subset | point/box prompt, multimask, IoU |
| 7 | Diffusion Policy | Push-T demonstrations | action diffusion, horizons, receding control |

## 데이터 다운로드 원칙

데이터 다운로드와 공부 코드는 분리한다.

```text
scripts/
→ 데이터/checkpoint 다운로드 및 환경 설정

practice/
→ 이미 준비된 데이터를 읽어서 논문 실습
```

`practice/*.py`는 자동 다운로드하지 않는다.

데이터가 이미 준비되어 있으면 다운로드 스크립트는 `[skip]` 하고 종료한다. 큰 데이터는 가능한 경우 이어받기(resume)를 사용한다.

전체 명령은 `docs/01_DATA_SETUP.md`에 정리되어 있다.

## 공부용 코드 원칙

- 논문에 실제 등장하는 dataset / benchmark / task를 사용한다.
- 논문 Figure와 코드의 1:1 대응을 우선한다.
- 과도한 factory / registry / dynamic import를 피한다.
- 반복이 조금 생겨도 stage / block / channel 수를 코드에 직접 보이게 한다.
- `forward()`는 논문의 data flow 순서대로 읽히게 한다.
- accuracy 한 숫자보다 feature / attention / prediction / error / failure case를 본다.
- 전체 재현이 어려우면 `Faithful / Scaled / Pretrained analysis`를 명확히 구분한다.

## 환경 설정

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
bash scripts/setup_env.sh
conda activate pytorch-dl-practice
python scripts/00_check_environment.py
```

이미 clone되어 있다면:

```bash
git pull
```

## 논문 한 편 완료 기준

다음 6가지를 자신의 말로 설명하고 최소 실습 1개를 완료한다.

```text
1. Problem
2. Core idea
3. Method
4. Input / GT / Output / Loss
5. Evidence
6. My observation
```

상세 체크리스트와 노트 템플릿은 `docs/03_PAPER_CHECKLIST.md`를 사용한다.

## 디렉터리 역할

```text
pytorch-deep-learning-practice/
├── docs/       # 학습 순서 / 데이터 준비 / 완료 체크리스트
├── lessons/    # 일반 딥러닝 기초
├── practice/   # 논문 단위 실습
├── scripts/    # 데이터 다운로드 / 환경 설정
├── outputs/    # 그림 / 로그 / checkpoint
├── external/   # official 외부 repository
├── AGENTS.md
└── README.md
```

코드가 실행된다는 것과 논문을 이해했다는 것은 같은 의미가 아니다. 실행 결과를 논문의 문제, 구조, 주장과 연결해서 설명할 수 있어야 한다.
