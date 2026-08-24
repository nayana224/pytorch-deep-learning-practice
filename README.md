# PyTorch Deep Learning Practice

오승상 딥러닝 강의의 핵심 개념을 **PyTorch로 직접 재현하고 Matplotlib으로 관찰**하기 위한 실습 저장소입니다.

이 저장소의 목표는 예제 코드를 많이 모으는 것이 아니라, 강의에서 배운 수식과 개념이 실제 학습 코드에서 어떻게 동작하는지 확인하는 것입니다.

## 현재 실습 범위

현재 강의 진도에 맞춰 다음 세 영역을 집중적으로 다룹니다.

- Deep Neural Network
- Convolutional Neural Network
- Recurrent Neural Network

강의 전체에는 이후 Attention, Auto-Encoder/VAE, GAN, NLP, GNN이 이어지며 해당 부분은 진도에 맞춰 순차적으로 추가합니다.

자세한 실습 의도와 관찰 포인트는 [`docs/STUDY_GUIDE.md`](docs/STUDY_GUIDE.md)를 먼저 읽는 것을 권장합니다.

## 환경

기본 Conda 환경:

- environment: `pytorch-dl-practice`
- Python: 3.11
- PyTorch: 2.12.1
- torchvision: 0.27.1

Linux + NVIDIA 환경에서는 PyTorch wheel이 필요한 CUDA runtime을 포함하므로 일반적인 PyTorch 실습만 한다면 시스템 CUDA Toolkit(`nvcc`)을 별도로 설치할 필요가 없습니다. NVIDIA driver는 필요합니다.

## 새 PC에서 시작

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
bash scripts/setup_env.sh
conda activate pytorch-dl-practice
python scripts/00_check_environment.py
```

연구실 PC에서는 현재 다음 경로를 사용합니다.

```text
~/inpyo_ws/pytorch-deep-learning-practice
```

코드는 절대 경로에 의존하지 않으므로 다른 노트북에서는 원하는 위치에 clone하면 됩니다.

## 추천 복습 순서

| 순서 | 실습 | 강의에서 확인할 핵심 | 대표 시각화 |
|---:|---|---|---|
| 1 | `00_xor_mlp.py` | Perceptron 한계, XOR, MLP, nonlinearity | linear vs nonlinear decision surface |
| 2 | `01_tensor_autograd.py` | chain rule, backpropagation, gradient | manual gradient 비교 |
| 3 | `02_linear_regression_manual.py` | cost, gradient descent, parameter update | 회귀선, loss, `w/b` 수렴 |
| 4 | `03_linear_regression_torch.py` | `nn.Module`, `backward`, optimizer | PyTorch 학습 과정 |
| 5 | `04_mnist_mlp.py` | MLP, logits, Softmax, Cross-Entropy | probability, confusion matrix, first-layer weights |
| 6 | `07_initialization_compare.py` | vanishing gradient, Xavier, He, ReLU | activation std, gradient flow |
| 7 | `06_optimizer_compare.py` | SGD, Momentum, Adagrad, RMSProp, Adam | optimizer loss curves |
| 8 | `10_batch_normalization.py` | Batch Normalization | layer별 activation mean/std |
| 9 | `08_regularization_compare.py` | overfitting, Dropout, L2 | train-test gap, decision boundary |
| 10 | `05_mnist_cnn.py` | convolution, pooling, feature map | learned filters, epoch별 feature maps |
| 11 | `09_rnn_sequence.py` | sequence, recurrent hidden state | hidden-state heatmap |

실행 예시:

```bash
python lessons/00_xor_mlp.py
python lessons/01_tensor_autograd.py
python lessons/02_linear_regression_manual.py
```

## Visualization-first 원칙

각 실습은 다음 순서로 공부합니다.

1. 실행 전에 입력/output tensor shape을 예상합니다.
2. 학습되는 parameter가 무엇인지 찾습니다.
3. loss에서 각 parameter로 gradient가 어떻게 전달되는지 생각합니다.
4. 코드를 실행합니다.
5. 터미널 값과 `outputs/`에 저장된 그림을 함께 봅니다.
6. 한 번에 조건 하나만 바꾸어 다시 실험합니다.

특히 CNN에서는 다음을 구분합니다.

```text
filter weight = optimizer가 업데이트하는 learnable parameter
feature map   = 입력과 현재 filter로 계산된 activation
```

따라서 feature map 자체가 parameter로 학습되는 것이 아니라 **filter가 학습됨에 따라 같은 입력의 feature map이 변화**합니다.

## 실습 구조

```text
pytorch-deep-learning-practice/
├── environment.yml
├── README.md
├── docs/
│   └── STUDY_GUIDE.md
├── scripts/
│   ├── 00_check_environment.py
│   └── setup_env.sh
├── lessons/
│   ├── 00_xor_mlp.py
│   ├── 01_tensor_autograd.py
│   ├── 02_linear_regression_manual.py
│   ├── 03_linear_regression_torch.py
│   ├── 04_mnist_mlp.py
│   ├── 05_mnist_cnn.py
│   ├── 06_optimizer_compare.py
│   ├── 07_initialization_compare.py
│   ├── 08_regularization_compare.py
│   ├── 09_rnn_sequence.py
│   └── 10_batch_normalization.py
└── outputs/
```

## 데이터

MNIST 실습은 `torchvision.datasets.MNIST`를 사용하며 처음 실행할 때 `./data`에 다운로드합니다.

## 결과 파일

각 실습은 결과를 다음처럼 분리하여 저장합니다.

```text
outputs/<lesson_name>/
```

예를 들어 CNN 실습:

```bash
python lessons/05_mnist_cnn.py
```

은 epoch별 Conv filter와 feature map, training curve를 `outputs/05_mnist_cnn/`에 저장합니다.

## 공부할 때 중요한 점

이 코드는 성능 benchmark가 목적이 아닙니다. 강의 개념을 눈으로 확인하기 위해 일부러 작은 network와 단순한 dataset을 사용합니다.

따라서 optimizer 비교에서 한 optimizer가 더 좋은 숫자를 보였다고 해서 일반적으로 더 우수하다고 결론 내리지 않고, initialization/regularization 실험도 해당 실험 조건에서 나타나는 현상을 이해하는 데 초점을 둡니다.
