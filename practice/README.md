# 논문 기반 practice

`practice/`는 논문 전체를 매번 처음부터 재현하는 곳이 아니다.

핵심 질문은 하나다.

> **이 논문의 가장 중요한 알고리즘/주장을 어떤 최소 실습으로 직접 확인할 수 있는가?**

## 공통 원칙

1. 첫 바퀴에서는 **폭을 우선**한다.
2. 모든 논문은 Level 1까지 읽는다.
3. 중요한 논문은 Level 2에서 핵심 mechanism 하나를 직접 본다.
4. 연구와 직접 연결되는 논문만 Level 3까지 간다.
5. 시각화는 많이 만드는 것이 아니라 **핵심 질문에 답하는 결정적인 그림만** 남긴다.
6. 생성 결과는 `outputs/<paper>/`에 저장한다.
7. toy example은 mechanism 이해용으로만 사용하고 evidence로 해석하지 않는다.
8. 실제 성능을 볼 때는 논문 dataset / official checkpoint / 공개 재현 모델을 우선한다.
9. 코드 주석은 핵심 연산의 **왜 필요한지 / 입력과 출력이 무엇인지 / 무엇을 확인해야 하는지**가 드러나도록 한글로 작성한다.

## 순서와 깊이

| 폴더 | 논문 | 깊이 |
|---|---|---|
| `01_resnet` | ResNet | Level 2 |
| `02_unet` | U-Net | Level 2~3 |
| `03_deeplabv3plus` | DeepLabv3+ | Level 2~3 |
| `04_attention_is_all_you_need` | Transformer | Level 2 |
| `05_vit` | ViT | Level 3 |
| `06_sam` | SAM | Level 3 |
| `07_dinov2` | DINOv2 | Level 3 |
| `08_act` | ACT | Level 3 |
| `09_ddpm` | DDPM | Level 2 |
| `10_diffusion_policy` | Diffusion Policy | Level 3 |

## 첫 바퀴 실행 방식

Level 2 논문은 가능하면 run-all 스크립트 하나로 핵심 그림을 연속 생성한다.

```bash
python practice/04_attention_is_all_you_need/07_run_all.py
python practice/08_act/04_run_all.py
python practice/09_ddpm/04_run_all.py
```

Level 3 논문은 dataset/checkpoint 준비가 필요하므로 README의 준비 단계와 분석 단계를 분리한다.

## README 공통 구조

가능하면 각 논문 README에 다음을 명시한다.

- Paper claim
- Study level
- Target configuration
- What to observe
- Run
- Outputs
- Paper vs practice

## 완료 기준

첫 바퀴에서 최소한 다음을 설명할 수 있으면 된다.

```text
Problem
Core idea
Method
Input / GT / Output / Loss
Evidence
핵심 mechanism visualization에서 직접 본 것
```

Level 3 논문에서는 여기에 prediction / failure case를 추가한다.
