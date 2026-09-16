# AGENTS.md

## 프로젝트 목적
이 저장소는 논문을 코드로 읽고 이해하기 위한 PyTorch 실습 저장소다. 목표는 소프트웨어 설계 기교가 아니라 **논문의 구조와 데이터 흐름이 코드에 그대로 보이게 하는 것**이다.

## practice 공통 원칙
- 반드시 해당 논문에 실제로 등장하는 dataset / benchmark / task를 사용한다.
- 논문에 없는 편의용 dataset을 기본 데이터로 대체하지 않는다.
- 논문이 사용한 input / GT / architecture / loss / augmentation / metric을 가능한 한 기준으로 삼는다.
- 전체 재현이 비현실적이면 `Faithful / Scaled / Pretrained analysis` 중 수준을 README에 명시한다.
- synthetic data는 shape 확인용으로만 사용한다.

## 데이터 다운로드 원칙
- `practice/`의 공부 코드는 데이터를 자동 다운로드하지 않는다. 공부 코드에서는 `download=False`를 기본으로 한다.
- 데이터 획득은 `scripts/`의 전용 다운로드 스크립트가 담당한다.
- 이미 필요한 데이터가 존재하면 다운로드를 **무조건 skip**한다.
- 큰 archive는 가능한 경우 `wget -c` 또는 `curl -C -`로 중단 지점부터 이어받는다.
- 다운로드가 중간에 끊겨도 완성된 데이터로 오인하지 않도록, archive 존재 여부가 아니라 실제 필요한 파일/디렉터리를 확인해 완료 여부를 판단한다.
- 공부 코드에서 데이터가 없으면 자동 다운로드하지 말고 실행해야 할 다운로드 스크립트를 명확히 안내한다.
- SA-1B처럼 라이선스 동의가 필요한 데이터는 자동 다운로드하지 않는다.

## 공부용 코드 작성 원칙
가장 중요한 원칙은 **논문 Figure를 코드에서 바로 읽을 수 있어야 한다**는 것이다.

- 과도한 factory, registry, dynamic import, metaprogramming을 쓰지 않는다.
- `SourceFileLoader` 같은 학습과 무관한 import 기교를 쓰지 않는다.
- DRY보다 논문과 코드의 1:1 대응성을 우선한다.
- 반복이 조금 생기더라도 stage, block, channel 수가 코드에 직접 보이게 한다.
- 한 줄에 여러 연산을 몰아 쓰지 않는다.
- 핵심 모델 파일은 `__init__`만 읽어도 전체 구조가 보이게 한다.
- `forward()`는 논문의 data flow 순서대로 위에서 아래로 읽히게 한다.
- helper 함수는 데이터 로딩, metric, 시각화처럼 모델 이해를 방해하지 않는 부분에만 쓴다.
- 공식 pretrained 모델을 쓰는 DINOv2/SAM 계열은 내부를 억지로 재구현하지 않고, `input → official model → feature/mask → analysis` 흐름을 명확히 보여준다.

## 시각화 우선 원칙
각 논문에서 다음을 가능한 한 직접 본다.
- input / GT
- tensor shape 흐름
- intermediate feature
- 논문 핵심 메커니즘
- prediction / probability
- metric curve
- error map / failure case

## 현재 practice
- `01_resnet`: CIFAR-10, plain vs residual, degradation/optimization
- `02_unet`: ISBI 2012 EM, valid conv, crop+concat
- `03_deeplabv3plus`: PASCAL VOC 2012, atrous conv, ASPP, decoder
- `04_vit`: 논문 downstream CIFAR-100, patch/token/attention
- `05_dinov2`: official DINOv2 + Oxford-IIIT Pets, frozen feature/PCA/probe
- `06_sam`: official SAM + SA-1B subset, point/box/multimask
- `07_diffusion_policy`: official Push-T demonstrations, action diffusion/receding horizon

## 논문 실습 완료 기준
1. Problem
2. Core idea
3. Method
4. Input / GT / Output / Loss
5. Evidence
6. My observation

새 작업마다 해당 README와 이 파일이 실제 저장소 상태와 맞는지 확인한다.
