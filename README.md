# PyTorch Deep Learning Practice

오승상 딥러닝 강의에서 배운 개념을 PyTorch 코드로 직접 확인하기 위한 순차 실습 저장소입니다.

## 목표

- 수식과 PyTorch 코드의 대응 관계를 이해합니다.
- `forward -> loss -> backward -> optimizer.step()` 학습 흐름을 직접 확인합니다.
- MLP, CNN, optimizer, initialization, regularization, RNN을 작은 실험으로 순서대로 익힙니다.
- 연구실 PC와 개인 노트북에서 같은 Conda 환경을 쉽게 재현합니다.

## 권장 환경

이 저장소의 기본 Conda 환경 이름은 `pytorch-dl-practice`입니다.

- Python: 3.11
- PyTorch: 2.12.1
- torchvision: 0.27.1

Linux + NVIDIA 환경에서는 PyTorch wheel이 필요한 CUDA runtime을 함께 제공하므로, 일반적인 PyTorch 실습만 한다면 시스템 CUDA Toolkit(`nvcc`)을 별도로 설치할 필요가 없습니다. NVIDIA driver는 필요합니다.

## 빠른 시작

Miniconda가 이미 설치되어 있다면:

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
bash scripts/setup_env.sh
conda activate pytorch-dl-practice
python scripts/00_check_environment.py
```

환경이 정상이라면 첫 실습부터 진행합니다.

```bash
python lessons/01_tensor_autograd.py
python lessons/02_linear_regression_manual.py
python lessons/03_linear_regression_torch.py
```

## 다른 PC에서 재현하기

새 PC에서는 원하는 경로에 저장소를 clone한 뒤 같은 setup script를 실행합니다.

```bash
git clone https://github.com/nayana224/pytorch-deep-learning-practice.git
cd pytorch-deep-learning-practice
bash scripts/setup_env.sh
conda activate pytorch-dl-practice
python scripts/00_check_environment.py
```

기존 환경을 `environment.yml`에 맞춰 갱신하려면 `setup_env.sh`를 다시 실행하면 됩니다.

## 연구실 PC 경로

현재 연구실 PC에서는 다음 경로를 사용합니다.

```text
~/inpyo_ws/pytorch-deep-learning-practice
```

저장소 내부 코드는 특정 절대 경로에 의존하지 않으므로 다른 노트북에서는 다른 위치에 clone해도 됩니다.

## 학습 순서

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

## 실습 원칙

1. 코드를 실행하기 전에 입력/출력 tensor shape을 먼저 예상합니다.
2. `forward -> loss -> backward -> optimizer.step()` 흐름을 매 실습에서 확인합니다.
3. 한 번에 하나의 조건만 바꾸어 결과를 비교합니다.
4. GPU 사용 여부와 tensor device를 명시적으로 확인합니다.
5. 실험 결과는 `outputs/`에 저장합니다.

## 데이터

MNIST 실습은 `torchvision.datasets.MNIST`를 사용하며 처음 실행할 때 `./data`에 다운로드합니다.
