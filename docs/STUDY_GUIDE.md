# 오승상 딥러닝 강의 PyTorch 복습 가이드

이 저장소는 강의 내용을 그대로 코드로 옮기는 것이 아니라, 핵심 개념을 **작은 실험으로 재현하고 시각적으로 확인**하는 데 목적이 있습니다.

현재 실습 범위는 강의의 다음 흐름에 맞춥니다.

1. Deep Neural Network
2. Convolutional Neural Network
3. Recurrent Neural Network

Attention 이후 내용은 강의 진도에 맞춰 순차적으로 추가합니다.

## 실습할 때 지킬 순서

각 파일을 실행하기 전에 다음 네 가지를 먼저 예상합니다.

1. 입력 tensor shape은 무엇인가?
2. 학습 parameter는 무엇인가?
3. loss가 어떤 parameter에 대한 gradient를 만드는가?
4. 실행 후 어떤 그림이 나오면 개념이 제대로 동작한 것인가?

실행 후에는 터미널 숫자보다 `outputs/`의 그림을 먼저 해석합니다.

---

## Part 1. DNN 기초

### 00. XOR와 MLP

```bash
python lessons/00_xor_mlp.py
```

관찰할 것:

- 단일 linear layer가 XOR을 제대로 분리하지 못하는 이유
- hidden layer와 nonlinear activation이 추가되면 decision boundary가 어떻게 달라지는가
- `single_layer_surface.png`와 `mlp_surface.png` 비교

핵심 질문:

> Linear layer를 여러 개 쌓아도 activation이 없다면 왜 결국 하나의 linear transformation과 같은가?

### 01. Tensor와 Autograd

```bash
python lessons/01_tensor_autograd.py
```

관찰할 것:

- `requires_grad=True`
- computation graph
- `backward()` 후 `.grad`
- 손으로 계산한 chain rule과 autograd 결과 비교

### 02. Gradient Descent 직접 구현

```bash
python lessons/02_linear_regression_manual.py
```

관찰할 것:

- MSE loss
- `grad_w`, `grad_b`
- learning rate에 따른 parameter update
- 회귀선이 target line으로 이동하는 과정

### 03. PyTorch 학습 loop

```bash
python lessons/03_linear_regression_torch.py
```

Lesson 02와 다음을 대응시킵니다.

```text
직접 gradient 계산       -> loss.backward()
직접 parameter update    -> optimizer.step()
gradient 초기화          -> optimizer.zero_grad()
```

### 04. MLP 분류, Softmax, Cross-Entropy

```bash
python lessons/04_mnist_mlp.py
```

관찰할 것:

- 입력 `[B, 1, 28, 28]`이 `Flatten`을 통해 `[B, 784]`가 되는 과정
- 마지막 layer의 출력은 probability가 아니라 **logits**라는 점
- `softmax(logits)`의 합이 1인지 확인
- `-log(p_true)`와 `CrossEntropyLoss`가 같은 값인지 확인
- confusion matrix와 예측 confidence
- 첫 hidden layer weight가 학습되며 만드는 공간 패턴

### 06. Optimizer 비교

```bash
python lessons/06_optimizer_compare.py
```

동일한 데이터와 동일한 초기 weight에서 다음만 바꿉니다.

- SGD
- Momentum
- Adagrad
- RMSprop
- Adam

관찰할 것:

- 초기 convergence 속도
- 최종 loss
- 같은 epoch 수에서 회귀 함수가 얼마나 잘 맞는지

중요: optimizer마다 최적 learning rate는 다를 수 있으므로, 이 실험은 절대적인 성능 순위가 아니라 **update 특성의 차이**를 보기 위한 실험입니다.

### 07. Vanishing Gradient와 Initialization

```bash
python lessons/07_initialization_compare.py
```

비교:

- Sigmoid + small normal
- Sigmoid + Xavier
- ReLU + large normal
- ReLU + He

관찰할 것:

- layer가 깊어질 때 activation standard deviation
- backward 시 layer별 gradient norm
- 앞쪽 layer gradient가 지나치게 작아지는지

`gradient_flow.png`가 핵심 결과입니다.

### 08. Regularization

```bash
python lessons/08_regularization_compare.py
```

작은 training set에 큰 MLP를 사용해 일부러 overfitting하기 쉬운 조건을 만든 뒤 다음을 비교합니다.

- no regularization
- L2 weight decay
- Dropout

관찰할 것:

- training accuracy와 test accuracy의 gap
- decision boundary가 지나치게 복잡해지는지
- regularization 후 경계가 어떻게 달라지는지

### 10. Batch Normalization

```bash
python lessons/10_batch_normalization.py
```

관찰할 것:

- BatchNorm 전후 layer별 activation mean/std
- `running_mean`, `running_var`
- learnable parameter `gamma(weight)`, `beta(bias)`

BatchNorm은 단순히 normalization 결과를 고정하는 것이 아니라, 정규화 후 learnable scale/shift를 적용합니다.

---

## Part 2. CNN

### 05. MNIST CNN과 Feature Map

```bash
python lessons/05_mnist_cnn.py
```

가장 중요한 실습입니다.

같은 test image 하나를 고정한 뒤 epoch별로 다음을 저장합니다.

- Conv1 filter weights
- Conv1 feature maps
- Conv2 feature maps
- training loss / test accuracy

구분:

```text
filter weight = 학습되는 parameter
feature map   = 현재 filter를 입력에 적용한 activation
```

따라서 feature map 자체가 parameter로 학습되는 것은 아닙니다. Filter가 학습되면서 동일한 입력에 대한 feature map이 달라집니다.

---

## Part 3. RNN

### 09. Sequence와 Hidden State

```bash
python lessons/09_rnn_sequence.py
```

관찰할 것:

- 입력 shape `[batch, sequence, feature]`
- 같은 값 집합이라도 순서가 label을 결정할 수 있음
- time step마다 hidden state가 어떻게 바뀌는지
- 마지막 hidden state를 classification에 사용하는 이유

`hidden_states_epoch_*.png`를 시간 방향으로 읽어보는 것이 핵심입니다.

---

## 추천 복습 순서

강의 복습용으로는 파일 번호 순서보다 아래 순서를 권장합니다.

```text
00 XOR
 -> 01 Autograd
 -> 02 Manual GD
 -> 03 PyTorch training loop
 -> 04 MLP + Softmax/CrossEntropy
 -> 07 Gradient flow + Xavier/He
 -> 06 Optimizers
 -> 10 BatchNorm
 -> 08 Regularization
 -> 05 CNN + feature maps
 -> 09 RNN + hidden state
```

이 순서는 `왜 DNN이 필요한가 -> 어떻게 학습하는가 -> 깊어졌을 때 무엇이 어려운가 -> 이미지와 sequence에 어떻게 확장하는가`의 흐름을 유지합니다.
