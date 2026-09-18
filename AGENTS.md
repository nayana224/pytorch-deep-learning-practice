# AGENTS.md

## 프로젝트 목적
이 저장소는 논문을 코드로 읽고 이해하기 위한 PyTorch 실습 저장소다. 목표는 소프트웨어 설계 기교가 아니라 **논문의 구조와 데이터 흐름이 코드에 그대로 보이게 하는 것**이다.

## practice 공통 원칙
- 반드시 해당 논문에 실제로 등장하는 dataset / benchmark / task를 사용한다.
- 논문에 없는 편의용 dataset을 기본 데이터로 대체하지 않는다.
- 논문이 사용한 input / GT / architecture / loss / augmentation / metric을 가능한 한 기준으로 삼는다.
- 전체 재현이 비현실적이면 `Faithful / Scaled / Pretrained analysis` 중 수준을 README에 명시한다.
- synthetic/toy data는 **Level 2의 메커니즘 확인용**으로만 사용할 수 있다.
- toy example에서 나온 결과를 논문 성능 evidence로 해석하지 않는다.
- Level 3의 성능/feature/failure 분석은 논문 dataset, official checkpoint, 또는 공개 재현 모델을 우선한다.

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

## 논문 학습 깊이 원칙

모든 논문을 같은 깊이로 구현하지 않는다.

- **Level 1 — Paper understanding**: 모든 논문. Problem / Core idea / Method / Input-GT-Output-Loss / Evidence를 설명할 수 있으면 된다.
- **Level 2 — Core mechanism check**: 핵심 연산 하나를 작은 tensor/example로 직접 확인한다. 전체 task training을 재현할 필요는 없다.
- **Level 3 — Model behavior analysis**: 연구와 직접 연결되는 논문만 실제 pretrained/faithful model을 돌려 feature, prediction, failure case를 본다.

첫 바퀴에서는 폭을 우선한다. 논문마다 시각화를 많이 만드는 대신 **핵심 주장 하나를 확인하는 결정적 visualization 1~3개**를 남긴다.

권장 깊이:
- ResNet: Level 2
- U-Net: Level 2~3
- DeepLabv3+: Level 2~3
- Attention Is All You Need: Level 2
- ViT: Level 3
- SAM: Level 3
- DINOv2: Level 3
- ACT: Level 3
- DDPM: Level 2
- Diffusion Policy: Level 3

## 시각화 우선 원칙
각 논문 실습의 목적은 단순히 학습 코드를 실행하는 것이 아니라 **논문의 핵심 주장이 실제로 관찰되는지 확인하는 것**이다.

각 논문에서 다음을 가능한 한 직접 본다.
- input / GT
- tensor shape 흐름
- intermediate feature
- 논문 핵심 메커니즘
- prediction / probability
- metric curve
- error map / failure case

논문 깊이에 따라 필요한 시각화 수를 다르게 한다.
- Level 2: 핵심 mechanism visualization 1~3개면 충분하다.
- Level 3: mechanism + prediction/feature + failure/evidence를 본다.

Level 3의 `practice/<paper>/`는 가능하면 아래 세 종류를 남긴다.
1. **Mechanism visualization** — 논문의 핵심 연산/feature/attention/noising/denoising 등이 실제로 어떻게 작동하는지
2. **Prediction visualization** — input / GT / prediction / probability 또는 trajectory
3. **Evidence visualization** — 논문의 핵심 주장과 직접 연결되는 비교, metric curve, ablation-like comparison, failure case

숫자 하나만 출력하고 끝내지 않는다. 결과 이미지는 `outputs/<paper>/`에 저장하여 나중에 논문 노트의 My observation 근거로 다시 볼 수 있게 한다.

`data/<paper>/`는 raw/official dataset과 필요한 sample input만 보관한다. 생성된 plot, feature map, prediction image를 `data/`에 섞지 않는다. 실습 결과는 모두 `outputs/<paper>/`에 모아 바로 확인한다.

## 모델 선택 원칙
- "논문에서 가장 좋은 모델"을 무조건 그대로 재학습하지 않는다.
- 먼저 논문 표에서 **핵심 주장을 가장 잘 대표하는 canonical/best reported configuration**을 식별한다.
- 공개 checkpoint와 현재 하드웨어로 현실적으로 실행 가능한 경우 그 모델을 우선 사용한다.
- 대규모 pretraining이 필요한 경우에는 공식 pretrained checkpoint로 분석한다.
- 절대 최고 성능이 multi-scale, ensemble, extra data, 비공개 데이터에 의존하면 그 조건을 README에 명확히 기록하고, 단일 모델 기준의 가장 강한 재현 가능한 설정을 실습 기준으로 삼는다.
- 모델 선택 이유를 각 README의 `Target configuration`에 한 줄로 명시한다.

## 현재/목표 practice 순서
- `01_resnet`: CIFAR-10, plain vs residual, degradation/optimization
- `02_unet`: ISBI 2012 EM, valid conv, crop+concat
- `03_deeplabv3plus`: PASCAL VOC 2012, atrous conv, ASPP, decoder
- `04_attention_is_all_you_need`: Transformer, Q/K/V, self/cross-attention, positional encoding
- `05_vit`: Vision Transformer, patch/token/attention
- `06_sam`: official SAM, point/box/multimask, promptable segmentation
- `07_dinov2`: official DINOv2, frozen feature/PCA/probe
- `08_act`: ACT, CVAE + Transformer, action chunking/temporal aggregation
- `09_ddpm`: DDPM, forward noising/reverse denoising/noise prediction
- `10_diffusion_policy`: official Push-T demonstrations, action diffusion/receding horizon

## 현재 표준 practice

`03_deeplabv3plus`는 Level 2~3 혼합형 reference이고, `04_attention_is_all_you_need`와 `09_ddpm`은 Level 2 최소 실습 reference로 사용한다.

이 폴더는 다음 세 층을 모두 포함해야 한다.
- mechanism: atrous sampling / ASPP branch / feature flow
- prediction: input / GT / probability / prediction / error / boundary error
- evidence: 동일한 scaled 조건의 no-decoder baseline과 decoder 모델 비교
- pretrained reality check: scratch 축소 모델의 낮은 성능을 논문 모델 성능으로 오해하지 않도록, 공개된 강한 pretrained model의 prediction/failure case를 별도 확인

다른 논문 폴더를 리팩터링할 때도 파일 개수를 기계적으로 맞추기보다 이 세 층과 README의 `Paper claim / Target configuration / What to observe / Outputs / Paper vs practice`를 우선 맞춘다.

## 논문 실습 완료 기준
1. Problem
2. Core idea
3. Method
4. Input / GT / Output / Loss
5. Evidence
6. My observation

새 작업마다 해당 README와 이 파일이 실제 저장소 상태와 맞는지 확인한다.
