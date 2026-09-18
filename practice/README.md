# 논문 기반 PyTorch 실습 로드맵

`practice/`는 이제 일반 딥러닝 문법 연습이 아니라 **논문을 읽고 실제 데이터/구조/loss/metric을 따라가며 검증하는 공간**이다.

MLP, CNN, Attention, Transformer 같은 기초 연산은 `lessons/`에서 다룬다. `practice/`는 논문 단위로 구성한다.

## 논문별 실습 깊이

모든 논문을 처음부터 끝까지 구현하지 않는다.

- **Level 1**: 논문 이해
- **Level 2**: 핵심 메커니즘 최소 실습
- **Level 3**: 실제 모델 feature / prediction / failure 분석

첫 바퀴의 목표는 각 논문의 핵심 주장과 알고리즘을 연결하는 것이다. Level 2에서는 전체 학습 파이프라인 대신 작은 tensor/example로 핵심 연산을 확인하고, Level 3에서만 pretrained 또는 논문에 가까운 모델을 실제 데이터에 적용한다.

생성 결과는 `data/`가 아니라 `outputs/<paper>/`에 저장한다. `data/`는 dataset/input 전용으로 유지한다.

## 공통 학습 방식
1. 논문에서 실제로 사용한 dataset / input / GT를 먼저 확인한다.
2. raw data → transform → tensor → batch 흐름을 직접 본다.
3. 논문 구조를 작은 단위의 실제 PyTorch 코드로 직접 타이핑한다.
4. model output과 GT가 loss에서 어떻게 만나는지 확인한다.
5. 논문 metric과 가능한 한 같은 기준으로 평가한다.
6. feature / prediction / failure case를 시각화한다.
7. 논문과 다른 축소 조건은 반드시 기록한다.

TODO 빈칸을 추측해서 채우는 방식은 사용하지 않는다. 대화에서 작은 실행 단위의 실제 코드를 받고 직접 타이핑한다.

## 현재 순서
1. `01_resnet/` — Deep Residual Learning for Image Recognition
2. `02_unet/` — U-Net
3. `03_deeplabv3plus/` — DeepLabv3+
4. `04_vit/` — Vision Transformer
5. `05_dinov2/` — DINOv2
6. `06_sam/` — Segment Anything
7. `07_diffusion_policy/` — Diffusion Policy

8번째 논문은 실제로 읽을 source가 확정되면 추가한다. 논문 없이 빈 디렉터리를 미리 만들지 않는다.

## 재현 수준
논문 재현은 세 단계로 구분한다.

- **Faithful**: dataset, architecture, loss, metric을 논문과 동일하게 사용
- **Scaled**: 핵심 조건은 유지하되 model size / epoch / subset 등을 줄임
- **Pretrained analysis**: 대규모 pretraining이 비현실적인 foundation model은 공식 pretrained model로 논문 주장을 분석

각 폴더 README에 현재 실습이 어느 수준인지 명시한다.

## 논문 1편 완료 기준
1. Problem
2. Core idea
3. Method
4. Input / GT / Output / Loss
5. Evidence
6. My observation

위 6가지를 자신의 말로 설명하고 최소 한 번 직접 실행해 관찰한 결과가 있어야 완료로 본다.
