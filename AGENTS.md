# AGENTS.md

## 프로젝트 목적
이 저장소는 PyTorch 딥러닝 학습용 실습 저장소다. 코드 완성보다 각 모델이 왜 필요한지, 데이터가 어떤 shape으로 흐르는지, 무엇을 관찰해야 하는지 이해하는 것을 우선한다.

사용자가 직접 코드를 한 단계씩 타이핑하며 공부하는 것이 핵심 목표다. AI/Codex는 사용자의 사고와 구현 연습을 대체하기보다, 실습의 순서와 관찰 포인트를 설계하는 역할을 우선한다.

## 작업 원칙
- 기존 `lessons/`의 MLP/CNN 실습은 보존한다.
- 논문/모델 확장 실습은 `notebooks/` 아래에 단계별로 추가한다.
- 한 실습은 하나의 핵심 질문을 가진다.
- 완성 코드를 처음부터 한꺼번에 넣기보다, 사용자가 셀 단위로 직접 타이핑하고 실행할 수 있는 작은 단계로 나눈다.
- 각 단계에는 `왜 필요한가`, `입력/출력 shape은 무엇인가`, `무엇을 관찰해야 하는가`를 명확히 적는다.
- accuracy만 기록하지 말고 tensor shape, activation, filter, feature map, attention, mask, error case 등 중간 결과를 적극적으로 시각화한다.
- 비교 실험에서는 한 번에 한 조건만 바꾼다.
- 결과를 보기 전에 가능한 경우 예상/가설을 먼저 적고, 실제 결과와 비교한다.
- 큰 foundation model(SAM/SAM2)은 처음부터 재구현하지 않고 공식 pretrained 모델을 사용한 inference와 내부 구조 분석을 우선한다.
- MLP, CNN, ResNet, U-Net, Attention, Transformer, ViT는 가능한 한 작은 형태를 직접 구현해 구조를 이해한다.
- 새 작업을 진행할 때 README와 본 파일의 범위/원칙이 현재 저장소 상태와 일치하는지 함께 확인한다.

## 디렉터리 정책
- `lessons/`: 기존 Python script 기반 기초 실습
- `notebooks/`: Jupyter Notebook 기반 논문/모델 실습
- `outputs/`: 실행 결과 이미지/모델/로그
- `scripts/`: 환경 설정과 점검 스크립트

## Notebook 작성 원칙
각 notebook은 가능한 한 짧은 셀로 나누고, 아래 순서를 따른다.

1. 제목 / 관련 논문
2. 핵심 질문
3. 실습 전 예상 또는 가설
4. 필요한 개념을 짧게 정리
5. 입력 데이터와 tensor shape 확인
6. 모델을 작은 모듈부터 직접 구현
7. forward 과정의 중간 shape 확인
8. 학습 또는 inference
9. 중간 표현 및 결과 시각화
10. 실패 사례 / 오분류 / 애매한 결과 확인
11. 결과 해석
12. 한 줄 정리
13. 다음 실습으로의 연결

## 단계별 구현 규칙
- 한 셀에서 너무 많은 개념을 동시에 구현하지 않는다.
- 가능하면 `데이터 확인 → 작은 모듈 → 전체 모델 → 학습 → 분석` 순서로 확장한다.
- 사용자가 직접 타이핑하는 단계에서는 boilerplate를 과도하게 늘리지 않는다.
- 핵심 연산은 라이브러리 한 줄로 숨기기 전에 한 번은 직접 구현해보는 것을 우선한다.
- 직접 구현 후 공식 구현이나 `torchvision` 구현과 비교하는 것은 권장한다.

## ResNet 논문 실습 진행 원칙
현재 ResNet 실습은 `notebooks/03_resnet/`에서 진행한다. 완성 코드를 미리 채우지 않고 사용자가 직접 타이핑하는 방식을 유지한다.

현재 진행 위치:
- `residual_block.ipynb`
- 1단계: 입력 tensor `x`의 shape / dtype / 값 범위를 확인한 뒤, 다음 셀부터 plain 2-layer block을 직접 타이핑한다.

실습 순서는 다음과 같다.

1. `residual_block.ipynb`
   - 입력 `x`의 shape 확인
   - plain 2-layer block을 직접 구현
   - residual branch `F(x)`를 직접 확인
   - identity shortcut으로 `x`를 더해 `F(x) + x`를 구성
   - `x`, `F(x)`, `F(x)+x`의 shape/value 통계 비교
   - 필요하면 projection shortcut을 추가하고 dimension mismatch를 직접 확인
   - 마지막에 간단한 backward를 수행하여 gradient가 실제로 계산되는지 확인

2. `resnet18_cifar10.ipynb`
   - CIFAR-10 input / GT / output / loss 흐름 확인
   - 직접 만든 `BasicBlock`으로 작은 Plain CNN과 작은 ResNet 구성
   - seed, optimizer, epoch, batch size 등 비교 조건을 통제
   - Plain CNN vs ResNet의 training loss / accuracy curve 비교
   - prediction 및 failure case 시각화
   - 가능하면 residual activation 크기 또는 gradient norm을 관찰

ResNet 실습의 핵심 질문은 다음 세 가지다.
- `H(x)`를 직접 근사하는 것과 `F(x)=H(x)-x`를 학습하는 것은 코드에서 어떻게 다른가?
- shortcut은 실제 tensor 연산에서 무엇을 하는가?
- 같은 조건에서 Plain CNN과 ResNet의 학습 양상이 실제로 어떻게 달라지는가?

논문의 152-layer ImageNet 결과를 그대로 재현하는 것이 목표가 아니다. 작은 실험으로 residual learning의 데이터 흐름과 optimization 차이를 관찰하는 것이 목표다.

## 시각화 우선 원칙
가능하면 다음 시각화를 포함한다.

### 공통
- input sample
- tensor shape 흐름
- training / validation loss
- accuracy 또는 task metric
- prediction 예시
- failure case

### MLP
- 입력 이미지와 flatten 결과
- 첫 layer weight를 이미지 형태로 재배치
- confusion matrix

### CNN
- convolution kernel/filter
- training 전/후 feature map
- layer별 activation
- receptive field를 이해할 수 있는 예시

### ResNet
- residual block 내부 shape
- `F(x)`와 `x`, `F(x)+x` 비교
- Plain CNN vs ResNet training curve
- 가능하면 gradient norm 또는 activation 분포 비교

### U-Net
- input image
- ground-truth mask
- predicted mask
- prediction overlay
- error map
- encoder/decoder feature map 일부

### Attention / Transformer
- Q, K, V shape
- attention score matrix
- softmax 전/후 값
- attention heatmap
- head별 attention 차이

### ViT
- 원본 이미지의 patch 분할
- patch embedding 전/후 shape
- positional embedding 적용 전/후
- attention map

### SAM / SAM2
- 입력 이미지/프레임
- point / box prompt
- predicted mask overlay
- prompt 변화에 따른 mask 비교
- SAM2에서는 frame별 propagation 결과와 memory 효과를 관찰할 수 있는 예시

## 출력 저장 규칙
- 생성 결과는 `outputs/` 아래 모델/실습별 폴더에 저장한다.
- notebook 화면에만 보여주지 말고 중요한 결과는 파일로도 저장한다.
- 파일명은 무엇을 보여주는지 알 수 있게 작성한다.

예:
- `outputs/03_resnet/plain_vs_resnet_loss.png`
- `outputs/04_unet/prediction_overlay.png`
- `outputs/05_attention/attention_heatmap.png`
- `outputs/07_vit/image_patches.png`

## 구현 우선순위
직접 구현:
- MLP
- CNN
- ResNet
- U-Net
- Attention
- Transformer
- ViT

공식 코드 사용 + 내부 분석:
- SAM
- SAM2

## 학습 결과 리뷰 원칙
- 실습 결과를 리뷰할 때는 `확인한 사실`과 `아직 검증하지 않은 가설/의문`을 분리한다.
- 데이터 파이프라인은 가능한 한 `raw data → transform → tensor → batch → model input` 순서로 추적하고, 각 단계의 shape, dtype, 수치 범위와 의미를 확인한다.
- `ToTensor()`의 0~1 스케일 변환, 입력 데이터의 normalization/standardization, 모델 내부의 BatchNorm/LayerNorm을 서로 다른 개념으로 구분한다.
- 단일 실행의 accuracy 차이만으로 모델의 일반적 우수성을 결론내리지 않는다. 비교 목적이면 seed, 데이터 순서, epoch, optimizer 등 통제 조건을 기록한다.
- 실제 연구 데이터로 확장하기 전에는 RGB/RGB-D의 채널 의미, 단위, invalid 값, 센서 정렬, 좌표계, normalization 기준, augmentation의 물리적 타당성을 별도 검증 대상으로 둔다.
- 전처리 비교 실험은 한 번에 하나의 조건만 바꾸고, 입력 분포와 학습/검증 결과 변화를 함께 기록한다.

## Custom Food와의 연결
후반 실습은 `custom_food_target_mass_ws`의 perception 문제와 연결할 수 있도록 설계한다.

예:
- U-Net 기반 food segmentation
- 기존 food mask와 SAM mask 비교
- RGB / Depth / mask 동시 시각화
- tray wall, sauce, reflection, depth invalid region 등의 failure case 기록
- 향후 graspable mask 또는 graspability map 학습 가능성 분석

단, foundation model을 프로젝트에 넣는 것 자체를 목표로 하지 않는다. 기존 pipeline에서 어떤 문제가 있고, 새 모델이 그 문제를 실제로 개선하는지 실험으로 검증한 뒤 적용 여부를 결정한다.
