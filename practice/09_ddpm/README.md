# 09. DDPM — Denoising Diffusion Probabilistic Models

실습 깊이: **Level 2 — Core mechanism check**

DDPM 전체 U-Net 학습과 고품질 image generation을 재현하지 않는다.
첫 바퀴에서는 논문의 핵심인 다음 세 가지를 직접 확인한다.

1. forward process에서 `x_0 -> x_t`로 noise가 점점 커진다.
2. training target은 clean image가 아니라 추가된 Gaussian noise `epsilon`이다.
3. `epsilon`을 잘 알면 noisy `x_t`에서 clean signal `x_0`를 추정할 수 있다.

논문이 CIFAR-10에서 실험하므로 실제 CIFAR-10 sample을 사용한다.

## 데이터

```bash
python scripts/download_torchvision_data.py ddpm
```

## 실행

```bash
python practice/09_ddpm/01_forward_noising.py
python practice/09_ddpm/02_noise_target.py
python practice/09_ddpm/03_reconstruct_x0.py
```

## 결과

```text
outputs/09_ddpm/
├── 01_forward_noising.png
├── 02_noise_prediction_target.png
└── 03_reconstruct_x0.png
```

## 이 실습에서 하지 않는 것

- DDPM U-Net 전체 학습
- 수백/수천 step의 실제 learned reverse sampling
- FID / Inception Score 재현
- 논문의 모든 variational bound 세부 구현

첫 바퀴에서는 아래 질문에 답할 수 있으면 충분하다.

> 왜 이미지에 noise를 넣는 forward process를 정의하고, 모델은 왜 epsilon을 예측하도록 학습하는가?
