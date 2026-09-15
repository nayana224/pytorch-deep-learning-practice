# AGENTS.md

## 프로젝트 목적
이 저장소는 PyTorch 딥러닝 학습용 실습 저장소다. 코드 완성 자체보다 각 모델이 왜 필요한지, 데이터가 어떤 shape으로 흐르는지, 무엇을 관찰해야 하는지 이해하는 것을 우선한다.

사용자가 직접 코드를 타이핑하며 공부하는 것이 핵심 목표다. AI/Codex는 대리 구현자가 아니라 실습 설계자, 코드 리뷰어, 디버깅 파트너 역할을 우선한다.

## 작업 원칙
- 기존 `lessons/`의 기초 실습은 보존한다.
- 논문/모델 확장 실습은 `practice/` 아래에 둔다.
- 저장소에서 직접 만드는 학습 실습은 기본적으로 `.py` 파일을 사용한다.
- 새 작업을 할 때마다 README와 본 파일이 현재 학습 방식과 일치하는지 확인하고 필요한 경우 함께 업데이트한다.
- 정확도만 보지 않고 tensor shape, activation, feature map, mask, attention, prediction, error case 등 중간 결과를 적극적으로 관찰한다.
- 비교 실험에서는 한 번에 한 조건만 바꾼다.
- 결과를 보기 전에 가능한 경우 예상/가설을 먼저 적고 실제 결과와 비교한다.
- 논문 실습은 가능하면 논문이 실제 사용한 데이터, 모델 구조, loss, augmentation, 평가 지표를 기준으로 한다. 단, 논문 재현에 불필요하거나 현재 환경에서 비현실적인 부분은 별도 축소 실험으로 분리하고 무엇을 바꿨는지 명확히 기록한다.
- 큰 foundation model(SAM/SAM2)은 처음부터 재구현하지 않고 공식 pretrained 모델 inference와 내부 구조 분석을 우선한다.
- MLP, CNN, ResNet, U-Net, Attention, Transformer, ViT는 가능한 한 작은 형태를 직접 구현하며 구조를 이해한다.

## 직접 타이핑 실습 방식
`practice/`의 기본 학습 방식은 TODO 빈칸 채우기가 아니다.

다음 사이클을 반복한다.

1. AI가 현재 단계에 필요한 작은 실행 단위의 실제 코드를 대화에서 제시한다.
2. 사용자가 그 코드를 직접 `.py` 파일에 타이핑한다.
3. 실행 전에 가능하면 tensor shape, 출력 의미, 변화 방향을 예상한다.
4. 사용자가 직접 실행한다.
5. 실행 결과와 이해되지 않는 줄을 바탕으로 질문한다.
6. AI가 결과를 해석하고 코드의 이유, failure mode, 다음 관찰 지점을 설명한다.
7. 이해가 끝나면 다음 작은 코드 조각으로 넘어간다.

원칙:
- 전체 완성 코드를 한 번에 던지지 않는다.
- 반대로 TODO만 남겨 사용자가 API 이름이나 문법을 추측하게 만들지도 않는다.
- 학습 핵심 연산은 실제 코드로 보여주되, 사용자가 직접 타이핑한다.
- 최종 파일에는 사용자가 단계적으로 직접 작성한 완성 코드가 남도록 한다.
- 질문이 생기면 다음 구현으로 넘어가기 전에 현재 데이터 흐름을 먼저 이해한다.
- 이미 충분히 이해한 boilerplate는 반복 설명하지 않는다.

## 공개/공식 코드 예외
- 외부 공식 구현이 있으면 먼저 언어, 프레임워크, 파일 구조, 논문과의 대응 관계를 확인한다.
- 공식 코드가 현재 학습 프레임워크와 다르면 원본을 보존하고 핵심 구조를 현재 프레임워크로 다시 구현해본다.
- 외부 공개 저장소의 `.ipynb`는 형식만 바꾸기 위해 임의로 `.py`로 변환하지 않는다.
- `external/` 등 외부 소스 디렉터리는 학습 편의를 위해 원본 코드를 임의 수정하지 않는다.

## 디렉터리 정책
- `lessons/`: 기존 Python script 기반 기초 실습
- `practice/`: 논문/모델별 직접 타이핑 실습 (`.py` 중심)
- `outputs/`: 실행 결과 이미지/모델/로그
- `scripts/`: 환경 설정과 점검 스크립트
- `external/`: 필요할 때 clone되는 공식/외부 소스. 원본 구조 보존

## U-Net 논문 실습 진행 원칙
현재 U-Net 실습은 `practice/04_unet/`에서 진행한다.

U-Net은 논문 Figure 1 구조뿐 아니라 논문의 실제 segmentation 실험 데이터 흐름도 가능한 한 그대로 따라간다. 기본 기준 데이터는 ISBI 2012 EM segmentation challenge 데이터다.

논문 기준 주요 사실:
- training data: 30장의 512x512 serial section transmission electron microscopy 이미지
- 각 training image에는 cell / membrane에 대한 fully annotated ground-truth segmentation map이 대응됨
- test segmentation map은 공개되지 않고 challenge server에서 평가
- 평가 지표: warping error, Rand error, pixel error
- architecture: valid 3x3 convolution, max pooling, up-convolution, crop + concat, final 1x1 convolution
- training: pixel-wise softmax + cross entropy, batch size 1, weighted loss, elastic deformation augmentation

실습 우선순위:
1. 원 논문 EM dataset을 확보하고 image / GT 구조를 먼저 확인한다.
2. raw image → tensor → GT mask의 shape, dtype, 값 범위를 확인한다.
3. 논문 Figure 1의 original U-Net 구조를 PyTorch로 직접 구현한다.
4. input tile / output crop 관계를 확인한다.
5. 기본 loss와 backward를 연결한다.
6. weighted loss와 elastic deformation은 기본 학습 흐름을 확인한 뒤 논문 순서대로 추가한다.
7. prediction과 GT를 시각화하고 논문 지표 또는 이해하기 쉬운 IoU/Dice도 보조 지표로 기록한다.

synthetic data는 더 이상 기본 경로가 아니다. 데이터셋 다운로드/전처리 문제와 모델 문제를 분리해야 할 때만 최소 진단용으로 사용한다.

## 시각화 우선 원칙
가능하면 input, GT mask, tensor shape 흐름, feature map, prediction, error map, failure case를 확인한다.

## 출력 저장 규칙
- 결과는 `outputs/` 아래 모델별 폴더에 저장한다.
- 중요한 결과는 재현 가능한 파일로 남긴다.
- 파일명만 보고 무엇을 보여주는지 알 수 있게 작성한다.

## 학습 결과 리뷰 원칙
- `확인한 사실`과 `아직 검증하지 않은 가설`을 분리한다.
- 데이터는 가능하면 `raw data → transform → tensor → batch → model input` 순서로 추적한다.
- 각 단계의 shape, dtype, 수치 범위, 의미를 확인한다.
- 논문과 다른 선택을 했다면 반드시 그 차이를 기록한다.
- 단일 실행의 accuracy 차이만으로 모델의 일반적 우수성을 결론내리지 않는다.
