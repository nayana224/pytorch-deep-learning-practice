# PyTorch Deep Learning Practice

논문을 **읽고 → 핵심 알고리즘을 직접 확인하고 → 필요한 논문만 깊게 실습**하기 위한 PyTorch 저장소다.

목표는 논문마다 전체 구현을 반복하는 것이 아니다.  
각 논문의 핵심 아이디어를 코드와 시각화로 연결하고, 실제 연구와 가까운 논문만 Level 3까지 내려간다.

## 공부 깊이

- **Level 1 — Paper understanding**  
  Problem / Core idea / Method / Input-GT-Output-Loss / Evidence를 설명한다.
- **Level 2 — Core mechanism check**  
  핵심 연산 하나를 작은 tensor 또는 실제 sample로 직접 확인한다.
- **Level 3 — Model behavior analysis**  
  official/pretrained/faithful model을 실제 데이터에 적용해 feature, prediction, failure case를 본다.

## 현재 순서

```text
practice/
├── 01_resnet/
├── 02_unet/
├── 03_deeplabv3plus/
├── 04_attention_is_all_you_need/
├── 05_vit/
├── 06_sam/
├── 07_dinov2/
├── 08_act/
├── 09_ddpm/
└── 10_diffusion_policy/
```

| 순서 | 논문 | 권장 깊이 | 첫 바퀴에서 직접 볼 것 |
|---:|---|---|---|
| 1 | ResNet | Level 2 | `x`, `F(x)`, `F(x)+x` |
| 2 | U-Net | Level 2~3 | crop/copy + decoder concat |
| 3 | DeepLabv3+ | Level 2~3 | atrous / ASPP / decoder boundary |
| 4 | Attention Is All You Need | Level 2 | Q/K/V → attention → mask → multi-head → positional/cross-attention |
| 5 | ViT | Level 3 | image → patch token → attention |
| 6 | SAM | Level 3 | prompt 변화에 따른 mask 변화 |
| 7 | DINOv2 | Level 3 | patch feature PCA / semantic retrieval |
| 8 | ACT | Level 3 | action chunk / temporal ensemble / CVAE |
| 9 | DDPM | Level 2 | forward noise / epsilon target / reconstruction |
| 10 | Diffusion Policy | Level 3 | noisy action → denoised action / receding horizon |

## 디렉터리 역할

```text
docs/       학습 순서 / 데이터 준비 / 완료 체크리스트
lessons/    일반 PyTorch / 딥러닝 기초
practice/   논문 단위 실습 코드
scripts/    dataset / checkpoint / 외부 repo 준비
data/       raw dataset / official input
outputs/    시각화 / metric / checkpoint
external/   official 외부 repository
```

**중요:** 생성된 그림은 `data/`에 넣지 않는다.  
`data/`는 입력 데이터 전용이고, 실습 결과는 모두 `outputs/<paper>/`에 저장한다.

## 기본 진행 순서

```text
1. 논문 읽기
2. practice/<paper>/README.md 확인
3. 핵심 mechanism script 실행
4. outputs/<paper>/ 그림 확인
5. 그림을 보며 논문의 핵심 주장을 내 말로 설명
6. 연구와 직접 연결되는 논문만 Level 3 진행
```

## 빠른 핵심 실습

전체 모델을 학습하지 않고 핵심 메커니즘만 바로 보고 싶을 때:

```bash
# Transformer attention 핵심
python practice/04_attention_is_all_you_need/07_run_all.py

# ACT 첫 바퀴 핵심
python practice/08_act/04_run_all.py

# DDPM 핵심
python scripts/download_torchvision_data.py ddpm
python practice/09_ddpm/04_run_all.py
```

상세 순서는 `docs/02_STUDY_ORDER.md`를 사용한다.
