# 04. 핵심 시각화 맵

첫 바퀴의 목표는 논문마다 많은 그림을 만드는 것이 아니다.

> **논문의 핵심 알고리즘/주장 하나를 직접 보고 설명할 수 있는가?**

각 폴더의 `00_run_core.py`를 실행하면 첫 바퀴에 필요한 핵심 실습만 순서대로 실행한다.

## 공통 실행

```bash
python practice/<paper>/00_run_core.py
```

생성 결과는 모두 다음 위치에서 확인한다.

```text
outputs/<paper>/
```

`data/<paper>/`에는 raw dataset / official input만 둔다.

---

## 논문별 핵심 관찰

| 순서 | 논문 | 깊이 | 첫 바퀴에서 반드시 볼 것 | 대표 output |
|---:|---|---|---|---|
| 1 | ResNet | Level 2 | `x → F(x) → F(x)+x` | `05_residual_mechanism.png` |
| 2 | U-Net | Level 2~3 | encoder crop + decoder upsample + concat | `02_model_crop_concat.png` |
| 3 | DeepLabv3+ | Level 2~3 | dilation sampling / ASPP / decoder flow | `02_atrous_receptive_field.png` |
| 4 | Attention Is All You Need | Level 2 | Q/K/V → score → softmax → mask → multi-head | `02_scaled_dot_product_attention.png` |
| 5 | ViT | Level 3 | image → 16×16 patch sequence → CLS/attention | `02_patches.png` |
| 6 | SAM | Level 3 | 같은 image에서 prompt가 바뀌면 mask가 어떻게 달라지는가 | `02_point.png` |
| 7 | DINOv2 | Level 3 | patch feature가 semantic region을 구분하는가 | `03_patch_pca.png` |
| 8 | ACT | Level 3 | single action이 아니라 chunk를 예측하고 temporal ensemble하는 이유 | `01_action_chunking.png` |
| 9 | DDPM | Level 2 | `x0 → xt`, epsilon target, x0 reconstruction | `01_forward_noising.png` |
| 10 | Diffusion Policy | Level 3 | image diffusion의 x가 action sequence로 바뀌는 과정 | `02_action_diffusion.png` |

---

## 첫 바퀴 완료 질문

각 논문에서 아래 네 문장에 답할 수 있으면 core 실습은 끝낸다.

```text
1. 이 그림에서 input은 무엇인가?
2. 논문의 핵심 연산이 그림의 어디에서 보이는가?
3. 연산 전후 tensor/feature/action이 어떻게 달라지는가?
4. 이 관찰이 논문의 핵심 주장과 어떻게 연결되는가?
```

첫 바퀴에서 답이 나오면 전체 모델 구현을 더 늘리지 않는다.
연구와 직접 연결되는 Level 3 논문만 두 번째 단계에서 pretrained/official model의 prediction과 failure case까지 본다.

---

## Level 2와 Level 3의 차이

### Level 2

작은 tensor 또는 최소 실제 sample로 **핵심 알고리즘 자체**를 확인한다.

예:

```text
Transformer:
Q/K/V → QK^T → scaling → softmax → V

DDPM:
x0 → noise 추가 → xt
epsilon target
epsilon을 알 때 x0 복원
```

전체 benchmark training은 필요 없다.

### Level 3

핵심 메커니즘에 더해 실제 model behavior를 본다.

```text
official/pretrained model
→ real input
→ feature / prediction
→ error / failure case
```

scratch 축소 모델의 낮은 성능을 논문 모델 성능으로 해석하지 않는다.
