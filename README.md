# PyTorch Deep Learning Practice

오승상 딥러닝 강의에서 배운 개념을 PyTorch 코드로 직접 확인하기 위한 순차 실습 저장소입니다.

## 학습 원칙

1. 코드를 실행하기 전에 입력/출력 tensor shape을 먼저 예상합니다.
2. `forward -> loss -> backward -> optimizer.step()` 흐름을 매 실습에서 확인합니다.
3. 한 번에 하나의 조건만 바꾸어 결과를 비교합니다.
4. GPU 사용 여부를 코드가 숨기지 않고 출력하도록 합니다.
5. 실습 결과는 `outputs/`에 저장합니다.

## 권장 학습 순서

| 순서 | 파일 | 핵심 개념 |
|---:|---|---|
| 0 | `scripts/00_check_environment.py` | Python, PyTorch, CUDA 환경 확인 |
| 1 | `lessons/01_tensor_autograd.py` | Tensor, shape, autograd |
| 2 | `lessons/02_linear_regression_manual.py` | gradient descent 수동 구현 |
| 3 | `lessons/03_linear_regression_torch.py` | `nn.Module`, loss, optimizer |
| 4 | `lessons/04_mnist_mlp.py` | DataLoader, MLP, 분류 학습 loop |
| 5 | `lessons/05_mnist_cnn.py` | Conv2d, ReLU, MaxPool2d |
| 6 | `lessons/06_optimizer_compare.py` | SGD, Momentum, Adam 비교 |
| 7 | `lessons/07_initialization_compare.py` | Xavier, He initialization |
| 8 | `lessons/08_regularization_compare.py` | Dropout, L2 weight decay |
| 9 | `lessons/09_rnn_sequence.py` | 순서가 있는 sample과 RNN |

## 환경 구성

PyTorch는 GPU/driver 조합에 따라 설치 명령이 달라지므로 `requirements.txt`에 고정하지 않았습니다.
먼저 연구실 PC의 환경을 확인한 뒤 공식 PyTorch 설치 명령을 선택합니다.

```bash
python3 --version
nvidia-smi
```

가상환경 예시:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

그 다음 환경에 맞는 PyTorch와 torchvision을 설치하고:

```bash
python -m pip install -r requirements.txt
python scripts/00_check_environment.py
```

## 데이터

MNIST 실습은 `torchvision.datasets.MNIST`를 사용하며 처음 실행할 때 `./data`에 다운로드합니다.

## 첫 실행

환경 확인이 끝나면 아래부터 순서대로 진행합니다.

```bash
python lessons/01_tensor_autograd.py
python lessons/02_linear_regression_manual.py
python lessons/03_linear_regression_torch.py
```

각 코드의 출력값을 읽고 다음 단계로 넘어가는 것을 권장합니다.
