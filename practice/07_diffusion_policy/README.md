# 07. Diffusion Policy — Visuomotor Policy Learning via Action Diffusion

이 폴더는 Diffusion Policy 논문의 action diffusion 데이터 흐름과 imitation learning 설정을 따라간다.

## 논문 기준
논문은 4개 robot manipulation benchmark의 15개 task에서 평가하며, observation-conditioned diffusion으로 action sequence를 생성한다.

핵심 요소:
- demonstration observation/action sequence
- conditional DDPM over actions
- observation horizon / prediction horizon / action execution horizon
- CNN-based 또는 Transformer-based noise prediction network
- receding-horizon control
- visual conditioning
- noise prediction MSE loss

전체 real-robot benchmark 재현은 장비/데이터 요구가 크다. 따라서 먼저 공개 benchmark/demo 데이터 중 재현 가능한 task(예: Push-T 계열)를 사용하고, 실제 로봇 실험과 차이는 명시한다.

## 진행 순서
1. `01_data.py`: demonstration의 observation/action sequence와 horizon 확인
2. `02_diffusion.py`: action에 noise를 넣고 timestep/noise target 확인
3. `03_policy.py`: observation-conditioned noise predictor 구현/사용
4. `04_train.py`: epsilon target과 MSE loss로 학습
5. `05_rollout.py`: denoising → action sequence → receding-horizon 실행
6. `06_analyze.py`: multimodality, action smoothness, failure rollout 분석

## 완료 기준
1. Problem: multimodal/high-dimensional sequential action을 안정적으로 예측하기 어려움
2. Core idea: policy를 conditional denoising diffusion process로 표현
3. Method: observation conditioning + action-sequence diffusion + receding horizon
4. Input / GT / Output / Loss: observation + noisy action + timestep → predicted noise, GT noise → MSE
5. Evidence: benchmark success rate와 ablation
6. My observation: denoising/action trajectory/failure rollout에서 직접 본 현상
