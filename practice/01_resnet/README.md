# 01. ResNet — Deep Residual Learning for Image Recognition

재현 수준: **CIFAR-10 실험을 중심으로 Faithful에 가까운 학습 실습**.

## 논문에서 볼 구조
```text
32x32 RGB
→ 3x3 conv, 16
→ stage1: 16ch residual block x3
→ stage2: 32ch residual block x3, 첫 block stride 2
→ stage3: 64ch residual block x3, 첫 block stride 2
→ global average pooling
→ FC 64 → 10
```

ResNet-20은 `6n+2`, `n=3`이다. shortcut은 CIFAR 실험의 option A를 사용한다.

## 파일
- `data.py`: CIFAR-10 + paper preprocessing
- `resnet.py`: PlainBlock / ResidualBlock / ResNet20
- `01_data.py`: raw image와 tensor 확인
- `02_model.py`: shape flow와 `F(x) + x` 확인
- `03_train.py`: paper SGD schedule
- `04_analyze.py`: plain/residual test error와 prediction
- `residual_block.py`: 이전 직접 타이핑 학습 이력 보존

코드는 소프트웨어 추상화보다 **논문 구조가 눈에 보이는 것**을 우선한다.
