# PyTorch Deep Learning Practice

MNIST 하나의 데이터셋을 충분히 뜯어본 뒤, MLP와 CNN만 사용해서 딥러닝의 핵심을 비교 실험하는 저장소입니다.

현재 단계에서는 범위를 넓히지 않습니다. RNN, optimizer 비교, initialization, regularization, batch normalization 등의 독립 lesson은 모두 제거하고 **MNIST 입력 데이터 → MLP → CNN → 입력 구조 실험** 순서에 집중합니다.

## 학습 목표

1. 모델보다 먼저 입력 데이터의 shape, dtype, range, class 분포와 pixel 통계를 확인합니다.
2. 같은 MNIST를 MLP와 CNN에 넣었을 때 입력을 처리하는 방식의 차이를 이해합니다.
3. accuracy만 보지 않고 weight, convolution filter, feature map을 Matplotlib으로 관찰합니다.
4. 한 번에 한 조건만 바꾸는 실험으로 결과의 원인을 해석합니다.

## 권장 순서

| 순서 | 코드 | 질문 |
|---:|---|---|
| 0 | `lessons/00_inspect_mnist.py` | 모델에 실제로 무엇을 넣고 있는가? |
| 1 | `lessons/01_mnist_mlp.py` | 28x28 이미지를 784차원 vector로 보면 무엇을 학습하는가? |
| 2 | `lessons/02_mnist_cnn.py` | 2차원 공간 구조를 보존하면 convolution은 무엇을 학습하는가? |
| 3 | `lessons/03_pixel_permutation_experiment.py` | pixel 값은 그대로 두고 위치만 섞으면 MLP와 CNN은 어떻게 달라지는가? |
| 4 | `lessons/04_input_corruption_experiment.py` | noise와 occlusion에 MLP와 CNN은 얼마나 강한가? |

## 0. MNIST 입력 데이터부터 뜯어보기

먼저 학습하지 않습니다.

```bash
python lessons/00_inspect_mnist.py
```

확인할 항목:

- train/test sample 개수
- 한 image tensor의 shape `[C, H, W] = [1, 28, 28]`
- dtype과 pixel range
- class별 sample 수
- 전체 training set의 pixel mean/std
- class별 실제 sample
- class별 평균 이미지
- pixel intensity 분포

생성 결과:

```text
outputs/00_inspect_mnist/
├── samples_by_class.png
├── class_distribution.png
├── class_mean_images.png
└── pixel_histogram.png
```

이 단계에서 가장 중요한 질문은 **"숫자 하나가 PyTorch 안에서는 어떤 tensor로 들어오는가?"** 입니다.

## 1. MLP baseline

```bash
python lessons/01_mnist_mlp.py
```

MLP는 입력 이미지를 먼저 flatten합니다.

```text
[B, 1, 28, 28]
        ↓ Flatten
[B, 784]
        ↓ Linear
[B, 128]
        ↓ ReLU
[B, 10]
```

관찰할 결과:

```text
outputs/01_mnist_mlp/
├── training_curves.png
├── first_layer_weights.png
├── confusion_matrix.png
└── test_predictions.png
```

`first_layer_weights.png`에서는 첫 hidden neuron의 784개 weight를 다시 28x28로 배치해서 어떤 pixel 패턴에 민감해졌는지 봅니다.

## 2. CNN baseline

```bash
python lessons/02_mnist_cnn.py
```

CNN은 flatten하기 전에 2차원 공간 구조를 유지합니다.

```text
[B, 1, 28, 28]
        ↓ Conv2d
[B, 16, 28, 28]
        ↓ Pool
[B, 16, 14, 14]
        ↓ Conv2d
[B, 32, 14, 14]
        ↓ Pool
[B, 32, 7, 7]
        ↓ Flatten + Linear
[B, 10]
```

관찰할 결과:

```text
outputs/02_mnist_cnn/
├── filters_before_training.png
├── filters_after_training.png
├── conv1_before_training.png
├── conv1_after_training.png
├── conv2_before_training.png
├── conv2_after_training.png
└── training_curves.png
```

여기서 구분할 것:

- **filter weight**: 실제로 gradient descent로 학습되는 parameter
- **feature map**: 현재 filter를 특정 입력에 적용해서 나온 activation

## 3. 핵심 실험: 공간 구조를 없애면?

```bash
python lessons/03_pixel_permutation_experiment.py
```

모든 MNIST 이미지에 **동일한 고정 pixel permutation**을 적용합니다. 각 image가 가진 784개의 pixel 값 자체는 보존하지만, 위/아래/옆 pixel의 관계는 무너집니다.

비교 조건:

```text
MLP + original MNIST
MLP + permuted MNIST
CNN + original MNIST
CNN + permuted MNIST
```

이 실험의 핵심은 CNN의 convolution이 왜 이미지의 **local spatial structure**를 전제로 하는지를 직접 확인하는 것입니다.

실험 전 가설:

- MLP는 모든 pixel을 fully connected 방식으로 보기 때문에 고정 permutation에 상대적으로 덜 민감할 수 있습니다.
- CNN은 가까운 pixel끼리의 지역적 패턴을 이용하므로 공간 구조가 무너지면 성능 저하가 더 클 것으로 예상합니다.

이 문장은 실험 가설이며, 실제 수치는 실행 결과로 판단합니다.

## 4. 입력 손상 실험

```bash
python lessons/04_input_corruption_experiment.py
```

clean MNIST로 학습한 MLP/CNN을 다음 test input에 그대로 평가합니다.

- clean image
- Gaussian noise가 추가된 image
- 중앙 10x10 영역이 가려진 image

이 실험에서는 **학습 조건은 그대로 두고 test input만 변경**합니다.

확인할 질문:

- 어느 입력 손상에서 accuracy가 가장 크게 떨어지는가?
- MLP와 CNN의 하락 폭은 같은가?
- 이미지의 local pattern을 이용하는 CNN의 특성이 robustness에도 도움이 되는가?

## 환경 구성

기존 Conda 환경을 그대로 사용합니다.

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
bash scripts/setup_env.sh
conda activate pytorch-dl-practice
python scripts/00_check_environment.py
```

연구실 PC에서는:

```text
~/inpyo_ws/pytorch-deep-learning-practice
```

를 사용합니다.

## 실험 원칙

- 입력 데이터를 먼저 확인합니다.
- baseline을 먼저 확보합니다.
- 한 번에 한 조건만 바꿉니다.
- 같은 비교에서는 random seed와 학습 조건을 가능한 한 맞춥니다.
- terminal accuracy만 보지 않고 Matplotlib 결과를 함께 봅니다.
- 결과를 보기 전에 먼저 가설을 적고, 결과가 예상과 다르면 그 이유를 찾습니다.

## 이후에 추가할 수 있는 실험

현재 코드에 바로 넣지는 않았습니다. 위 네 단계를 충분히 본 뒤 필요하면 다음 순서로 확장하는 것이 좋습니다.

1. training data 양: 1k / 10k / 60k
2. normalization 유무
3. model capacity: hidden units / channel 수
4. optimizer: SGD vs Adam
5. regularization: Dropout / weight decay

먼저 **MNIST 자체와 MLP/CNN의 차이를 이해한 뒤** 이 변수들을 추가합니다.
