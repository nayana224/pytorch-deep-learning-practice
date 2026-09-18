# 논문 실습 문서

이 폴더는 논문을 **같은 깊이로 전부 구현하지 않고**, 목적에 따라 Level 1/2/3으로 나누어 공부하기 위한 가이드다.

## 읽는 순서

1. `01_DATA_SETUP.md` — 필요한 dataset / checkpoint 준비
2. `02_STUDY_ORDER.md` — 첫 바퀴에서 실행할 최소 실습
3. `03_PAPER_CHECKLIST.md` — 논문을 읽었다 / 실습했다의 기준 점검

## 전체 순서

```text
01 ResNet
02 U-Net
03 DeepLabv3+
04 Attention Is All You Need
05 ViT
06 SAM
07 DINOv2
08 ACT
09 DDPM
10 Diffusion Policy
```

## 학습 깊이

```text
Level 1: 논문 이해
Level 2: 핵심 메커니즘 최소 실습
Level 3: 실제 모델 feature / prediction / failure 분석
```

첫 바퀴에서는 논문마다 결정적인 visualization 1~3개만 남겨도 된다.

## 파일 역할

```text
data/      raw dataset / official input
outputs/   실습 결과 png / json / checkpoint
practice/  논문별 코드
scripts/   다운로드 / 외부 repo / checkpoint 준비
```

toy example은 Level 2 메커니즘 설명에는 사용할 수 있지만, 논문 성능 evidence로 해석하지 않는다.
