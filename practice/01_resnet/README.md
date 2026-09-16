# 01. ResNet — Deep Residual Learning for Image Recognition

재현 수준: **CIFAR-10 experiment에 대해 Faithful에 가까운 로컬 재현**.

논문은 CIFAR-10에서 32×32 입력, per-pixel mean subtraction, 3×3 conv, `{16,32,64}` channels, `6n+2` layers, option-A identity shortcut을 사용한다. 학습은 SGD, momentum 0.9, weight decay 1e-4, batch 128, lr 0.1에서 시작해 32k/48k iteration에 10배씩 감소하고 64k에 종료한다. augmentation은 4-pixel padding 후 random 32×32 crop과 horizontal flip이다.

## 파일
- `cifar_resnet.py`: paper CIFAR plain/residual network, option A shortcut
- `01_data.py`: CIFAR-10 download, per-pixel mean subtraction, paper augmentation 확인
- `02_model.py`: 20/56-layer shape와 parameter 수 확인
- `03_train.py`: 논문 iteration schedule로 plain/residual 학습
- `04_analyze.py`: test error curve, prediction, feature response magnitude 관찰
- `residual_block.py`: 이전 직접 타이핑 학습 이력 보존

## 실행
```bash
python practice/01_resnet/01_data.py
python practice/01_resnet/02_model.py
python practice/01_resnet/03_train.py --kind plain --depth 20
python practice/01_resnet/03_train.py --kind resnet --depth 20
python practice/01_resnet/03_train.py --kind resnet --depth 56
python practice/01_resnet/04_analyze.py
```

핵심 비교는 **같은 depth의 plain vs residual**, 그리고 **depth 증가 시 degradation/optimization**이다. 논문은 CIFAR-10에서 ResNet-20/32/44/56/110을 비교한다.
