# 02. 논문 실습 순서

첫 바퀴에서는 **논문 전체 구현보다 핵심 메커니즘 확인**을 우선한다.

## 01. ResNet — Level 2

```bash
python scripts/download_torchvision_data.py cifar10
python practice/01_resnet/02_model.py
python practice/01_resnet/05_residual_mechanism.py
```

핵심 질문:

```text
x, F(x), F(x)+x는 실제 tensor에서 어떻게 연결되는가?
plain network와 residual network는 무엇이 다른가?
```

## 02. U-Net — Level 2~3

```bash
bash scripts/download_isbi2012.sh
python practice/02_unet/02_model.py
python practice/02_unet/02_model.py
```

핵심 질문:

```text
encoder feature가 crop/copy되어 decoder feature와 어디에서 concat되는가?
```

## 03. DeepLabv3+ — Level 2~3

메커니즘:

```bash
bash scripts/download_voc2012.sh
python practice/03_deeplabv3plus/02_atrous.py
python practice/03_deeplabv3plus/03_model.py
```

scaled model 분석:

```bash
python practice/03_deeplabv3plus/04_train.py --variant v3plus --epochs 20
python practice/03_deeplabv3plus/05_analyze.py
```

실제로 잘 학습된 prediction 확인:

```bash
bash scripts/setup_deeplabv3plus_pretrained.sh
python practice/03_deeplabv3plus/07_pretrained_analyze.py
```

핵심 질문:

```text
dilation rate는 sampling 범위를 어떻게 바꾸는가?
ASPP branch는 왜 여러 개 필요한가?
low-level feature가 decoder boundary refinement에 왜 필요한가?
```

## 04. Attention Is All You Need — Level 2

```bash
python practice/04_attention_is_all_you_need/05_run_all.py
```

핵심 질문:

```text
Q/K/V는 무엇인가?
QK^T -> scaling -> softmax -> V가 어떤 의미인가?
mask는 왜 미래 token을 막는가?
여러 head는 왜 다른 attention pattern을 만들 수 있는가?
```

## 05. ViT — Level 3

첫 바퀴:

```bash
python scripts/download_torchvision_data.py cifar100
python practice/05_vit/01_data.py
python practice/05_vit/02_patches.py
python practice/05_vit/03_model.py
```

두 번째 단계:

```bash
python practice/05_vit/04_train.py --model tiny
python practice/05_vit/05_analyze.py
```

핵심 질문:

```text
이미지가 어떻게 patch token sequence가 되는가?
CLS token / position embedding / attention은 어디에 들어가는가?
```

## 06. SAM — Level 3

```bash
bash scripts/setup_sam.sh
bash scripts/download_sam_vit_b.sh

python practice/06_sam/02_point_prompt.py
python practice/06_sam/03_box_prompt.py
python practice/06_sam/04_ambiguity.py
```

핵심 질문:

```text
같은 image embedding에 prompt가 달라지면 mask가 어떻게 달라지는가?
single point가 ambiguous할 때 multimask output이 왜 필요한가?
```

## 07. DINOv2 — Level 3

```bash
python scripts/download_torchvision_data.py pets
python practice/07_dinov2/02_features.py
python practice/07_dinov2/03_pca.py
python practice/07_dinov2/05_analyze.py
```

핵심 질문:

```text
label 없이 학습한 patch feature가 semantic structure를 실제로 갖는가?
비슷한 이미지가 frozen representation에서 가까워지는가?
```

## 08. ACT — Level 3

첫 바퀴 mechanism check:

```bash
python practice/08_act/01_action_chunking.py
python practice/08_act/02_temporal_ensemble.py
python practice/08_act/03_cvae_latent.py
```

핵심 질문:

```text
왜 single action 대신 action chunk를 예측하는가?
겹치는 chunk prediction을 temporal ensemble하는 이유는 무엇인가?
CVAE latent z는 어떤 역할을 하는가?
```

현재 코드는 toy mechanism visualization이다. 실제 ALOHA model/dataset analysis는 2차 Level 3 실습에서 추가한다.

## 09. DDPM — Level 2

```bash
python scripts/download_torchvision_data.py ddpm
python practice/09_ddpm/01_forward_noising.py
python practice/09_ddpm/02_noise_target.py
python practice/09_ddpm/03_reconstruct_x0.py
```

핵심 질문:

```text
x0는 timestep이 증가하면서 어떻게 noise가 되는가?
왜 model target이 x0가 아니라 epsilon인가?
epsilon을 알면 왜 clean signal을 다시 추정할 수 있는가?
```

## 10. Diffusion Policy — Level 3

```bash
bash scripts/setup_diffusion_policy.sh
bash scripts/download_pusht.sh

python practice/10_diffusion_policy/01_data.py
python practice/10_diffusion_policy/02_diffusion.py
python practice/10_diffusion_policy/03_policy.py
python practice/10_diffusion_policy/05_rollout.py
python practice/10_diffusion_policy/06_analyze.py
```

필요할 때만 scaled training:

```bash
python practice/10_diffusion_policy/04_train.py --epochs 10
```

핵심 질문:

```text
DDPM의 x가 robot action sequence로 바뀌면 무엇이 달라지는가?
observation conditioning은 어디에 들어가는가?
prediction/action/observation horizon은 어떻게 다른가?
receding horizon으로 어떤 action만 실제 실행하는가?
```

## 첫 바퀴 종료 기준

각 논문마다 다음을 모두 구현할 필요는 없다.

```text
[ ] Problem을 설명할 수 있다.
[ ] Core idea를 설명할 수 있다.
[ ] Method의 큰 data flow를 설명할 수 있다.
[ ] Input / GT / Output / Loss를 설명할 수 있다.
[ ] 핵심 experiment 하나가 무엇을 검증하는지 설명할 수 있다.
[ ] outputs/<paper>/의 핵심 그림 하나를 보고 내가 관찰한 것을 말할 수 있다.
```
