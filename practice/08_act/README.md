# 08. ACT — Action Chunking with Transformers

목표 깊이: **Level 3**  
현재 폴더의 즉시 실행 가능한 부분: **Level 2 core mechanism check**

ACT 논문에서 가장 먼저 체감해야 할 것은 전체 ALOHA 시스템 재현이 아니라 다음 세 가지다.

1. single action이 아니라 **action chunk**를 예측한다.
2. 여러 시점의 겹치는 chunk prediction을 **temporal ensemble**로 합친다.
3. CVAE latent `z`를 사용해 demonstration의 multimodality를 모델링한다.

## 실행

```bash
python practice/08_act/01_action_chunking.py
python practice/08_act/02_temporal_ensemble.py
python practice/08_act/03_cvae_latent.py
```

## 결과

```text
outputs/08_act/
├── 01_action_chunking.png
├── 02_temporal_ensemble.png
└── 03_cvae_latent.png
```

## 주의

현재 세 스크립트는 **논문 알고리즘을 눈으로 이해하기 위한 toy mechanism visualization**이다.
ALOHA dataset/model의 실제 성능 evidence가 아니다.

ACT는 연구 연결성이 높은 Level 3 논문이므로 2차 실습에서는 official ACT/ALOHA code와 demonstration을 사용해 다음을 추가한다.

- image/joint observation
- GT action chunk
- predicted action chunk
- temporal aggregation 결과
- 성공/실패 rollout

첫 바퀴에서는 위 세 그림을 보고 다음 질문에 답할 수 있으면 충분하다.

> 왜 action 하나가 아니라 chunk를 예측하고, 여러 chunk prediction을 다시 temporal ensemble하는가?
