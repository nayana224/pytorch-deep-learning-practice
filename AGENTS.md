# AGENTS.md

## 프로젝트 목적
이 저장소는 PyTorch 딥러닝 학습용 실습 저장소다. 코드 완성보다 각 모델이 왜 필요한지, 데이터가 어떤 shape으로 흐르는지, 무엇을 관찰해야 하는지 이해하는 것을 우선한다.

## 작업 원칙
- 기존 `lessons/`의 MLP/CNN 실습은 보존한다.
- 논문/모델 확장 실습은 `notebooks/` 아래에 단계별로 추가한다.
- 한 실습은 하나의 핵심 질문을 가진다.
- accuracy만 기록하지 말고 tensor shape, feature/attention/mask 등 중간 결과를 시각화한다.
- 비교 실험에서는 한 번에 한 조건만 바꾼다.
- 큰 foundation model(SAM/SAM2)은 처음부터 재구현하지 않고 공식 pretrained 모델을 사용한 inference와 내부 구조 분석을 우선한다.
- MLP, CNN, ResNet, U-Net, Attention, Transformer, ViT는 가능한 한 작은 형태를 직접 구현해 구조를 이해한다.
- 새 작업을 진행할 때 README와 본 파일의 범위/원칙이 현재 저장소 상태와 일치하는지 함께 확인한다.

## 디렉터리 정책
- `lessons/`: 기존 Python script 기반 기초 실습
- `notebooks/`: Jupyter Notebook 기반 논문/모델 실습
- `outputs/`: 실행 결과 이미지/모델/로그
- `scripts/`: 환경 설정과 점검 스크립트

## Notebook 기본 구성
각 notebook은 가능하면 다음 순서를 따른다.
1. 학습 목표 / 핵심 질문
2. 입력 데이터와 tensor shape 확인
3. 모델 구성
4. forward 과정의 중간 shape 확인
5. 학습 또는 inference
6. 시각화
7. 결과 해석
8. 다음 실험에서 바꿀 한 가지 변수
