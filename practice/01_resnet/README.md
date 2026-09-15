# 01. ResNet — Deep Residual Learning for Image Recognition

이 폴더는 ResNet 논문을 PyTorch로 따라가며 이해하는 실습이다.

## 원칙
- 논문에서 실제로 사용한 문제/데이터/구조/비교를 우선한다.
- 사용자는 대화에서 받은 실제 코드를 직접 타이핑한다.
- TODO 빈칸 채우기보다 작은 실행 단위로 코드를 누적한다.
- 논문 전체 성능 재현이 너무 비싸면, 무엇을 줄였는지 명시하고 핵심 주장만 작은 실험으로 검증한다.

## 논문 기준 실습
논문은 ImageNet과 CIFAR-10에서 plain network와 residual network를 비교한다. 로컬 실습은 비용을 고려해 CIFAR-10 실험을 우선한다.

논문 CIFAR-10 실험에서 확인할 핵심:
- 32x32 RGB 입력
- 10 classes
- plain network vs residual network
- depth가 증가했을 때 degradation이 나타나는지
- shortcut이 optimization을 어떻게 바꾸는지

## 파일
- `residual_block.py`: 이미 직접 타이핑한 residual block 학습 이력 보존
- `01_data.py`: CIFAR-10 데이터와 preprocessing 확인
- `02_model.py`: 논문 CIFAR용 plain/residual architecture 구현
- `03_train.py`: 동일 조건 학습 및 비교
- `04_analyze.py`: training error, test error, prediction, failure case 분석

## 완료 기준
1. Problem: 깊은 plain network의 degradation problem
2. Core idea: H(x)를 직접 학습하지 않고 F(x)=H(x)-x를 학습
3. Method: identity/projection shortcut과 residual block
4. Input / GT / Output / Loss: CIFAR-10 image → logits → CE loss
5. Evidence: plain vs residual의 training/test error 비교
6. My observation: 직접 본 activation/gradient/prediction 차이
