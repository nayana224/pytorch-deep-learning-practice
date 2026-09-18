# 02. 논문 실습 순서

첫 바퀴에서는 **논문 전체 구현보다 핵심 메커니즘 확인**을 우선한다.

모든 논문 폴더는 같은 형태로 시작한다.

```bash
python practice/<paper>/00_run_core.py
```

이 명령은 전체 training을 자동으로 돌리지 않는다.
README에 정의된 첫 바퀴 핵심 실습만 실행하고, 결과는 `outputs/<paper>/`에 저장한다.

---

## 01. ResNet — Level 2

준비:

```bash
python scripts/download_torchvision_data.py cifar10
```

첫 바퀴:

```bash
python practice/01_resnet/00_run_core.py
```

핵심 질문:

```text
x, F(x), shortcut x, F(x)+x는 실제 tensor에서 어떻게 연결되는가?
residual addition이 없다면 block의 data flow는 어떻게 달라지는가?
```

전체 CIFAR-10 training/degradation 비교는 두 번째 단계다.

---

## 02. U-Net — Level 2~3

준비:

```bash
bash scripts/download_isbi2012.sh
```

첫 바퀴:

```bash
python practice/02_unet/00_run_core.py
```

핵심 질문:

```text
valid convolution 때문에 spatial size가 왜 줄어드는가?
encoder feature를 왜 crop하는가?
crop한 encoder feature와 upsampled decoder feature는 어디에서 concat되는가?
```

prediction / probability / failure case는 필요할 때 `03_train.py → 04_analyze.py`로 본다.

---

## 03. DeepLabv3+ — Level 2~3

준비:

```bash
bash scripts/download_voc2012.sh
```

첫 바퀴:

```bash
python practice/03_deeplabv3plus/00_run_core.py
```

핵심 질문:

```text
dilation rate가 커지면 sampling 위치가 어떻게 벌어지는가?
ASPP는 왜 여러 dilation branch를 동시에 사용하는가?
low-level feature는 decoder에서 무엇을 보완하는가?
```

실제 prediction 품질은 scratch scaled model과 분리해서 본다.

```bash
bash scripts/setup_deeplabv3plus_pretrained.sh
python practice/03_deeplabv3plus/07_pretrained_analyze.py
```

---

## 04. Attention Is All You Need — Level 2

첫 바퀴:

```bash
python practice/04_attention_is_all_you_need/00_run_core.py
```

핵심 질문:

```text
Q/K/V는 같은 input에서 왜 서로 다른 projection을 사용하는가?
QK^T → scaling → softmax → V는 각각 무엇을 의미하는가?
causal mask는 왜 미래 token의 weight를 0으로 만드는가?
multi-head는 왜 여러 attention pattern을 병렬로 보는가?
positional encoding은 왜 필요한가?
cross-attention에서 Q와 K/V는 어디에서 오는가?
```

WMT 번역 전체 학습은 첫 바퀴에서 하지 않는다.

---

## 05. Vision Transformer — Level 3

준비:

```bash
python scripts/download_torchvision_data.py cifar100
```

첫 바퀴:

```bash
python practice/05_vit/00_run_core.py
```

핵심 질문:

```text
이미지가 어떻게 16x16 patch sequence가 되는가?
patch embedding 뒤에 CLS token과 position embedding은 어떻게 붙는가?
Transformer attention이 image patch 사이의 관계를 어떻게 다루는가?
```

scratch tiny training은 구조 이해 이후 선택적으로 한다.

---

## 06. SAM — Level 3

준비:

```bash
bash scripts/setup_sam.sh
bash scripts/download_sam_vit_b.sh
# SA-1B 공식 subset은 라이선스 동의 후 data/06_sam/sa1b/에 둔다.
```

첫 바퀴:

```bash
python practice/06_sam/00_run_core.py
```

이 runner는 SAM을 한 번만 load하고 같은 image embedding에서 point / box / multimask를 연속 확인한다.

핵심 질문:

```text
같은 image embedding에서 point와 box prompt가 mask를 어떻게 바꾸는가?
single point가 ambiguous할 때 왜 여러 mask를 출력하는가?
predicted IoU와 actual IoU는 어떻게 다른가?
```

---

## 07. DINOv2 — Level 3

준비:

```bash
bash scripts/setup_dinov2.sh
python scripts/download_torchvision_data.py pets
```

첫 바퀴:

```bash
python practice/07_dinov2/00_run_core.py
```

핵심 질문:

```text
pretrained CLS feature와 patch feature는 무엇이 다른가?
label 없이 학습한 patch feature가 semantic region을 구분하는가?
비슷한 이미지가 frozen representation에서 가까워지는가?
```

DINOv2 pretraining 자체는 재현하지 않는다.

---

## 08. ACT — Level 3

첫 바퀴:

```bash
python practice/08_act/00_run_core.py
```

핵심 질문:

```text
왜 single action 대신 future action chunk를 한 번에 예측하는가?
서로 다른 시점에서 나온 overlapping chunk를 왜 temporal ensemble하는가?
CVAE latent z는 demonstration variation을 어떻게 표현하는가?
```

현재 첫 바퀴 코드는 toy mechanism visualization이다.
실제 ALOHA model/dataset 분석은 2차 Level 3 단계다.

---

## 09. DDPM — Level 2

준비:

```bash
python scripts/download_torchvision_data.py ddpm
```

첫 바퀴:

```bash
python practice/09_ddpm/00_run_core.py
```

핵심 질문:

```text
x0는 timestep이 증가하면서 어떻게 xt가 되는가?
왜 학습 target이 clean image가 아니라 epsilon인가?
epsilon을 안다고 가정하면 x0를 어떻게 다시 추정할 수 있는가?
```

전체 U-Net generation training은 첫 바퀴에서 하지 않는다.

---

## 10. Diffusion Policy — Level 3

준비:

```bash
bash scripts/setup_diffusion_policy.sh
bash scripts/download_pusht.sh
```

첫 바퀴:

```bash
python practice/10_diffusion_policy/00_run_core.py
```

핵심 질문:

```text
DDPM의 image x가 robot action sequence로 바뀌면 무엇이 달라지는가?
observation은 diffusion model에 어떤 condition으로 들어가는가?
prediction / observation / action horizon은 어떻게 다른가?
```

scaled training과 rollout은 두 번째 단계다.

```bash
python practice/10_diffusion_policy/04_train.py --epochs 10
python practice/10_diffusion_policy/05_rollout.py
python practice/10_diffusion_policy/06_analyze.py
```

---

## 첫 바퀴 종료 기준

각 논문마다 아래에 답할 수 있으면 다음 논문으로 넘어간다.

```text
[ ] Problem을 내 말로 설명할 수 있다.
[ ] Core idea를 내 말로 설명할 수 있다.
[ ] Method의 큰 data flow를 설명할 수 있다.
[ ] Input / GT / Output / Loss를 설명할 수 있다.
[ ] 핵심 experiment 하나가 무엇을 검증하는지 설명할 수 있다.
[ ] outputs/<paper>/의 핵심 그림 하나를 보고 내가 관찰한 것을 말할 수 있다.
```

전체 시각화 맵은 `docs/04_CORE_VISUALIZATION_MAP.md`를 참고한다.
