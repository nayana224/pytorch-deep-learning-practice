# PyTorch Deep Learning Practice

오승상 딥러닝 강의에서 배운 개념을 PyTorch 코드로 직접 확인하기 위한 순차 실습 저장소입니다.

## 목표

- 수식과 PyTorch 코드의 대응 관계를 이해합니다.
- `forward -> loss -> backward -> optimizer.step()` 학습 흐름을 직접 확인합니다.
- MLP, CNN, optimizer, initialization, regularization, RNN을 작은 실험으로 순서대로 익힙니다.
- 터미널 숫자만 보는 것이 아니라 Matplotlib으로 학습 과정과 내부 표현을 시각적으로 확인합니다.
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

| 순서 | 파일 | 핵심 개념 | 대표 시각화 |
|---:|---|---|---|
| 0 | `scripts/00_check_environment.py` | Python, PyTorch, CUDA 환경 확인 | CUDA tensor 출력 |
| 1 | `lessons/01_tensor_autograd.py` | Tensor, shape, autograd | 수치 gradient 비교 |
| 2 | `lessons/02_linear_regression_manual.py` | gradient descent 수동 구현 | 회귀선 변화, loss, `w/b` 수렴 |
| 3 | `lessons/03_linear_regression_torch.py` | `nn.Module`, loss, optimizer | PyTorch 회귀선, parameter 수렴 |
| 4 | `lessons/04_mnist_mlp.py` | DataLoader, MLP, 분류 학습 loop | loss/accuracy, 첫 layer weight, prediction |
| 5 | `lessons/05_mnist_cnn.py` | Conv2d, ReLU, MaxPool2d | epoch별 feature map, learned filter, loss/accuracy |
| 6 | `lessons/06_optimizer_compare.py` | SGD, Momentum, Adam 비교 | optimizer별 loss curve |
| 7 | `lessons/07_initialization_compare.py` | Xavier, He initialization | layer별 activation 표준편차 |
| 8 | `lessons/08_regularization_compare.py` | Dropout, L2 weight decay | dropout mask, activation 분포 |
| 9 | `lessons/09_rnn_sequence.py` | 순서가 있는 sample과 RNN | sequence, hidden-state heatmap, loss/accuracy |

## Visualization-first 실습 원칙

1. 코드를 실행하기 전에 입력/출력 tensor shape을 먼저 예상합니다.
2. `forward -> loss -> backward -> optimizer.step()` 흐름을 매 실습에서 확인합니다.
3. 한 번에 하나의 조건만 바꾸어 결과를 비교합니다.
4. GPU 사용 여부와 tensor device를 명시적으로 확인합니다.
5. 가능한 실습은 학습 전/중/후 상태를 Matplotlib으로 저장합니다.
6. CNN에서는 동일한 입력 이미지를 고정하고 epoch별 feature map 변화를 비교합니다.
7. 단순 accuracy만 보지 않고 weight, activation, hidden state처럼 내부 표현도 관찰합니다.
8. 실험 결과는 lesson별 `outputs/` 하위 디렉터리에 저장합니다.

## 시각화 결과 확인

예를 들어 CNN 실습을 실행하면:

```bash
python lessons/05_mnist_cnn.py
```

다음과 같은 파일이 생성됩니다.

```text
outputs/05_mnist_cnn/
├── conv1_feature_maps_epoch_00.png
├── conv1_feature_maps_epoch_01.png
├── ...
├── conv2_feature_maps_epoch_00.png
├── conv2_feature_maps_epoch_01.png
├── ...
├── conv1_filters_epoch_00.png
├── conv1_filters_epoch_01.png
└── training_curves.png
```

여기서 가장 중요한 비교는 **같은 MNIST 입력 한 장에 대해 epoch 0, 1, 2, ...의 feature map이 어떻게 달라지는지**입니다. 초기에는 거의 임의의 반응을 보이지만 학습이 진행되면서 서로 다른 channel이 획, 경계, 국소 패턴 등에 선택적으로 반응하는 모습을 관찰할 수 있습니다. 다만 각 channel에 사람이 정한 의미가 자동으로 부여되는 것은 아니므로, feature map은 정성적 해석 도구로 사용합니다.

## 데이터

MNIST 실습은 `torchvision.datasets.MNIST`를 사용하며 처음 실행할 때 `./data`에 다운로드합니다.
