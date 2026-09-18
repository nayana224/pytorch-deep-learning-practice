# 07. Diffusion Policy — Visuomotor Policy Learning via Action Diffusion

재현 수준: **Official-code Push-T pipeline + scaled local training**.

논문은 4 benchmarks / 15 tasks에서 평가하며, 공개 코드의 대표 visual task인 **Push-T**를 실제 demonstration dataset으로 사용한다. 이 폴더는 다른 toy trajectory를 만들지 않고 공식 `pusht.zip`의 `pusht_cchi_v7_replay.zarr`를 사용한다.

공식 image Push-T 설정을 그대로 읽어온 핵심 값:
- image `[3,96,96]`, agent position `[2]`, action `[2]`
- prediction horizon 16
- observation horizon 2
- execution/action horizon 8
- DDPM train timesteps 100, `squaredcos_cap_v2`, epsilon prediction
- ConditionalUNet1D: step embedding 128, down dims `[512,1024,2048]`, kernel 5, groups 8
- visual crop 84×84, global observation conditioning
- AdamW `lr=1e-4`, betas `(0.95,0.999)`, weight decay `1e-6`
- official full config trains 3050 epochs with EMA/cosine/warmup; `04_train.py` defaults to only 10 epochs and is therefore **Scaled**, not a performance reproduction.

## 준비
```bash
bash scripts/setup_diffusion_policy.sh
bash scripts/download_pusht.sh
```
Full official environment dependencies are non-trivial; use `external/diffusion_policy/README.md` environment instructions. Practice code imports the official dataset, image policy, ConditionalUNet1D, normalizer, and DDPM behavior rather than rewriting them inaccurately.

## 파일/실행
```bash
python practice/10_diffusion_policy/01_data.py
python practice/10_diffusion_policy/02_diffusion.py
python practice/10_diffusion_policy/03_policy.py
python practice/10_diffusion_policy/04_train.py --epochs 10
python practice/10_diffusion_policy/05_rollout.py
python practice/10_diffusion_policy/06_analyze.py
```

- `01_data`: actual Push-T image / agent_pos / 16-step action chunk
- `02_diffusion`: action forward diffusion and epsilon target visualization
- `03_policy`: official image-conditioned policy의 실제 `compute_loss` 한 번 통과
- `04_train`: official policy/data/loss를 사용한 scaled local optimization
- `05_rollout`: 16-step prediction에서 다음 8 action을 실행하는 receding-horizon slicing 시각화
- `06_analyze`: 같은 observation에서 stochastic samples 여러 개를 뽑아 multimodality/spread를 관찰

실제 environment success-rate rollout과 논문 수치 비교는 official workspace의 PushTImageRunner로 full checkpoint를 평가하는 fidelity 단계다.
