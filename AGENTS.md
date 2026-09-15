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
- 큰 foundation model(SAM/SAM2)은 처음부터 재구현하지 않고 공식 pretrained 모델 inference와 내부 구조 분석을 우선한다.
- MLP, CNN, ResNet, U-Net, Attention, Transformer, ViT는 가능한 한 작은 형태를 직접 구현하며 구조를 이해한다.

## 직접 타이핑 실습 방식
`practice/`의 기본 학습 방식은 TODO 빈칸 채우기가 아니다.

다음 사이클을 반복한다.

1. AI가 현재 단계에 필요한 **작은 실행 단위의 실제 코드**를 대화에서 제시한다.
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

## 현재 논문/모델 실습 구조

```text
practice/
├── 01_mlp/
│   └── mnist_mlp.py
├── 02_cnn/
│   ├── mnist_cnn.py
│   └── feature_maps.py
├── 03_resnet/
│   ├── residual_block.py
│   └── resnet18_cifar10.py
├── 04_unet/
│   ├── README.md
│   ├── unet_architecture.py
│   └── segmentation.py
├── 05_attention/
│   ├── single_head_attention.py
│   └── multi_head_attention.py
├── 06_transformer/
│   └── transformer_encoder.py
├── 07_vit/
│   ├── patch_embedding.py
│   └── vit_cifar10.py
├── 08_sam/
│   ├── sam_image.py
│   └── sam_food.py
└── 09_sam2/
    ├── sam2_image.py
    └── sam2_video.py
```

기존 ResNet 파일은 학습 이력을 보존한다. 새로운 방식으로 전환하기 위해 과거 코드를 불필요하게 초기화하지 않는다. 이후 실습부터 작은 실제 코드 제시 → 직접 타이핑 → 실행 → 질문 방식으로 진행한다.

## U-Net 논문 실습 진행 원칙
현재 U-Net 실습은 `practice/04_unet/`에서 진행하며 새로운 직접 타이핑 방식을 우선 적용한다.

### `unet_architecture.py`
논문 Figure 1의 original U-Net tensor 흐름을 PyTorch로 직접 재구성한다.

진행 순서:
- 입력 `[1, 1, 572, 572]`
- valid `3x3 Conv + ReLU` 두 번
- max pooling
- contracting path
- bottleneck
- up-convolution
- encoder feature center crop
- channel-wise concatenation
- expanding path
- 마지막 `1x1 Conv`
- segmentation logits

각 단계에서 실제 코드가 대화로 제공되며 사용자가 직접 타이핑한다. 특히 `572 → 570 → 568`, pooling 전후 shape, skip feature, crop 전후 shape, concat 전후 channel 수를 직접 출력해 확인한다.

### `segmentation.py`
구조를 이해한 뒤 데이터 흐름을 연결한다.

진행 순서:
- synthetic image / GT mask
- Dataset / DataLoader
- model logits
- loss
- backward / optimizer step
- 작은 sample overfit sanity check
- probability / binary prediction
- IoU / Dice
- prediction / error map / failure case

기본 pipeline을 이해하기 전에는 elastic deformation, touching-cell weighted loss, 실제 biomedical dataset, modern padding U-Net 변형을 동시에 추가하지 않는다. 이후 한 번에 하나씩 확장한다.

U-Net의 핵심 질문:
- contracting path는 spatial detail을 줄이면서 어떤 context를 얻는가?
- decoder만으로 localization을 충분히 복원하기 어려운 이유는 무엇인가?
- encoder high-resolution feature가 무엇을 보완하는가?
- ResNet의 element-wise addition과 U-Net의 channel-wise concatenation은 어떻게 다른가?
- original U-Net에서 valid convolution 때문에 crop이 왜 필요한가?
- 마지막 `1x1 Conv`는 각 pixel의 feature vector를 무엇으로 바꾸는가?

## 시각화 우선 원칙
가능하면 다음을 확인한다.

### 공통
- input sample
- tensor shape 흐름
- training / validation loss
- task metric
- prediction 예시
- failure case

### CNN / ResNet / U-Net
- convolution filter 또는 feature map
- layer/stage별 activation 또는 shape
- ResNet: `F(x)`, `x`, `F(x)+x`
- U-Net: input, GT mask, predicted mask, overlay, error map, encoder/decoder feature

### Attention / Transformer / ViT
- Q, K, V shape
- attention score / softmax
- attention heatmap
- ViT patch 분할과 embedding shape
- positional embedding 전후

### SAM / SAM2
- point / box prompt
- predicted mask overlay
- prompt 변화에 따른 mask 차이
- SAM2 frame propagation과 memory 효과

## 출력 저장 규칙
- 결과는 `outputs/` 아래 모델별 폴더에 저장한다.
- 중요한 결과는 재현 가능한 파일로 남긴다.
- 파일명만 보고 무엇을 보여주는지 알 수 있게 작성한다.

예:
- `outputs/03_resnet/plain_vs_resnet_loss.png`
- `outputs/04_unet/prediction_overlay.png`
- `outputs/05_attention/attention_heatmap.png`
- `outputs/07_vit/image_patches.png`

## 학습 결과 리뷰 원칙
- `확인한 사실`과 `아직 검증하지 않은 가설`을 분리한다.
- 데이터는 가능하면 `raw data → transform → tensor → batch → model input` 순서로 추적한다.
- 각 단계의 shape, dtype, 수치 범위, 의미를 확인한다.
- `ToTensor()` 스케일 변환, 입력 normalization/standardization, 모델 내부 BatchNorm/LayerNorm을 서로 다른 개념으로 구분한다.
- 단일 실행의 accuracy 차이만으로 모델의 일반적 우수성을 결론내리지 않는다.
- 비교 목적이면 seed, 데이터 순서, epoch, optimizer 등 통제 조건을 기록한다.
- 실제 연구 데이터로 확장하기 전에는 RGB/RGB-D 채널 의미, 단위, invalid 값, 센서 정렬, 좌표계, normalization 기준, augmentation의 물리적 타당성을 별도로 검증한다.

## Custom Food와의 연결
후반 실습은 `custom_food_target_mass_ws`의 perception 문제와 연결할 수 있도록 설계한다.

예:
- U-Net 기반 food segmentation
- 기존 food mask와 SAM mask 비교
- RGB / Depth / mask 동시 시각화
- tray wall, sauce, reflection, depth invalid region failure case 기록
- 향후 graspable mask 또는 graspability map 학습 가능성 분석

foundation model을 프로젝트에 넣는 것 자체가 목표는 아니다. 기존 pipeline의 실제 문제와 새 모델의 개선 여부를 실험으로 검증한 뒤 적용 여부를 결정한다.
