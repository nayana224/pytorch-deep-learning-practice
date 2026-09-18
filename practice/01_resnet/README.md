# 01. ResNet — Deep Residual Learning for Image Recognition

실습 깊이: **Level 2**\n\n재현 수준: **CIFAR-10 실험을 중심으로 Faithful에 가까운 학습 실습**.


## 공통 첫 바퀴 실행

이 폴더의 핵심 실습만 연속 실행하려면:

```bash
python practice/01_resnet/00_run_core.py
```

전체 training을 자동으로 수행하는 명령이 아니라, 첫 바퀴에서 봐야 할 핵심 메커니즘만 실행한다.
생성된 그림은 `outputs/01_resnet/`에서 확인한다.

## 데이터 준비
공부 코드는 데이터를 자동 다운로드하지 않는다. 먼저 한 번만 실행한다.

```bash
python scripts/download_torchvision_data.py cifar10
```

이미 CIFAR-10이 있으면 `[skip]`하고 다시 받지 않는다.

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
- `04_analyze.py`: plain/residual test error와 prediction\n- `05_residual_mechanism.py`: `x`, `F(x)`, shortcut, `F(x)+x` feature map 시각화
- `residual_block.py`: 이전 직접 타이핑 학습 이력 보존

코드는 소프트웨어 추상화보다 **논문 구조가 눈에 보이는 것**을 우선한다.


## 첫 바퀴 권장

```bash
python practice/01_resnet/02_model.py
python practice/01_resnet/05_residual_mechanism.py
```

첫 바퀴에서는 전체 CIFAR-10 학습보다 **residual addition 자체를 이해하는 것**을 우선한다.
학습 curve와 degradation 비교는 두 번째 단계에서 `03_train.py`, `04_analyze.py`로 확인한다.
