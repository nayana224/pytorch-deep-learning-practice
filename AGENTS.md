# AGENTS.md

## 프로젝트 목적

이 저장소는 논문을 코드로 읽고 이해하기 위한 PyTorch 실습 저장소다. 목표는 소프트웨어 설계 기교가 아니라 **논문의 구조와 데이터 흐름이 코드에 그대로 보이게 하는 것**이다.

## practice 공통 원칙

- 반드시 해당 논문에 실제로 등장하는 dataset / benchmark / task를 사용한다.
- 논문에 없는 편의용 dataset을 기본 데이터로 대체하지 않는다.
- 논문이 사용한 input / GT / architecture / loss / augmentation / metric을 가능한 한 기준으로 삼는다.
- 전체 재현이 비현실적이면 `Faithful / Scaled / Pretrained analysis` 중 수준을 README에 명시한다.
- synthetic/toy data는 **Level 2 메커니즘 확인용**으로만 사용할 수 있다.
- toy example 결과를 논문 성능 evidence로 해석하지 않는다.
- Level 3 분석은 논문 dataset, official checkpoint, 또는 공개 재현 모델을 우선한다.

## 데이터 다운로드 원칙

- `practice/` 코드는 데이터를 자동 다운로드하지 않는다. 공부 코드에서는 `download=False`를 기본으로 한다.
- 데이터 획득은 `scripts/` 전용 다운로드 스크립트가 담당한다.
- 이미 필요한 데이터가 존재하면 다운로드를 skip한다.
- 큰 archive는 가능한 경우 이어받기를 지원한다.
- 라이선스 동의가 필요한 데이터는 자동 다운로드하지 않는다.

## 공부용 코드 작성 원칙

가장 중요한 원칙은 **논문 Figure와 핵심 수식을 코드에서 바로 읽을 수 있어야 한다**는 것이다.

- 과도한 factory, registry, dynamic import, metaprogramming을 쓰지 않는다.
- DRY보다 논문과 코드의 1:1 대응성을 우선한다.
- 반복이 조금 생기더라도 stage, block, channel/token 수가 코드에 직접 보이게 한다.
- 한 줄에 여러 핵심 연산을 몰아 쓰지 않는다.
- `forward()`는 논문의 data flow 순서대로 읽히게 한다.
- helper 함수는 데이터 로딩, metric, 시각화처럼 모델 이해를 방해하지 않는 부분에만 쓴다.
- 공식 pretrained model을 쓰는 DINOv2/SAM 계열은 내부를 억지로 재구현하지 않는다.
- 주석은 가능한 한 한글로 작성한다.
- 주석은 단순 코드 번역보다 **왜 필요한 연산인지 / 입력과 출력이 무엇인지 / 논문에서 무엇을 확인하는지**를 설명한다.

## 논문 학습 깊이 원칙

모든 논문을 같은 깊이로 구현하지 않는다.

- **Level 1 — Paper understanding**: Problem / Core idea / Method / Input-GT-Output-Loss / Evidence
- **Level 2 — Core mechanism check**: 핵심 연산을 작은 tensor/example로 직접 확인
- **Level 3 — Model behavior analysis**: 실제 pretrained/faithful model의 feature, prediction, failure case 분석

첫 바퀴에서는 폭을 우선한다. 논문마다 시각화를 많이 만드는 대신 **핵심 주장에 답하는 결정적 visualization**만 남긴다.

권장 깊이:

- ResNet: Level 2
- U-Net: Level 2~3
- DeepLabv3+: Level 2~3
- Attention Is All You Need: Level 2 — Q/K/V, scaled attention, mask, multi-head, positional encoding, cross-attention
- ViT: Level 3
- SAM: Level 3
- DINOv2: Level 3
- ACT: Level 3
- DDPM: Level 2
- Diffusion Policy: Level 3

## 시각화 원칙

Level 2:
- 핵심 mechanism visualization만 있으면 된다.
- 전체 task training은 필수가 아니다.
- 가능하면 run-all script 하나로 결과를 한 번에 생성한다.

Level 3:
1. Mechanism visualization
2. Prediction / feature visualization
3. Evidence / failure visualization

결과는 모두 `outputs/<paper>/`에 저장한다.

`data/<paper>/`는 raw/official dataset과 필요한 input만 보관하고, 생성된 plot이나 prediction image를 섞지 않는다.

## 모델 선택 원칙

- "논문에서 가장 좋은 숫자"를 무조건 그대로 재학습하지 않는다.
- 핵심 주장을 대표하는 canonical/best reported configuration을 먼저 식별한다.
- 공개 checkpoint와 현재 하드웨어로 실행 가능한 경우 그 모델을 우선 사용한다.
- 대규모 pretraining이 필요한 경우 official pretrained checkpoint로 분석한다.
- ensemble / extra data / 비공개 데이터가 필요한 최고 수치는 README에 조건을 명시한다.

## 현재 practice 순서

- `01_resnet`
- `02_unet`
- `03_deeplabv3plus`
- `04_attention_is_all_you_need`
- `05_vit`
- `06_sam`
- `07_dinov2`
- `08_act`
- `09_ddpm`
- `10_diffusion_policy`

## reference practice

- `03_deeplabv3plus`: Level 2~3 혼합형 reference
- `04_attention_is_all_you_need`: Level 2 수식/attention reference
- `09_ddpm`: Level 2 diffusion reference

다른 폴더를 리팩터링할 때 파일 개수를 기계적으로 맞추지 않는다. 대신 **무엇을 관찰해야 논문 핵심을 이해했다고 말할 수 있는가**를 기준으로 구성한다.

## 논문 실습 완료 기준

1. Problem
2. Core idea
3. Method
4. Input / GT / Output / Loss
5. Evidence
6. My observation

새 작업마다 해당 README와 이 파일이 실제 저장소 상태와 맞는지 확인한다.
