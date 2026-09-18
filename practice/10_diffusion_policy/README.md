# 10. Diffusion Policy — Visuomotor Policy Learning via Action Diffusion

실습 깊이: **Level 3**

재현 수준: **Official-code Push-T pipeline + scaled local training**

논문은 4 benchmarks / 15 tasks에서 평가하며, 공개 코드의 대표 visual task인 **Push-T**를 실제 demonstration dataset으로 사용한다. 이 폴더는 toy trajectory를 기본 데이터로 대체하지 않고 공식 `pusht_cchi_v7_replay.zarr`를 사용한다.

공식 image Push-T 설정에서 읽어온 핵심 값:

- image `[3,96,96]`, agent position `[2]`, action `[2]`
- prediction horizon 16
- observation horizon 2
- execution/action horizon 8
- DDPM train timesteps 100, `squaredcos_cap_v2`, epsilon prediction
- ConditionalUNet1D: step embedding 128, down dims `[512,1024,2048]`
- visual crop 84×84, global observation conditioning
- `04_train.py` 기본 10 epoch는 **Scaled**이며 performance reproduction이 아니다.


## 공통 첫 바퀴 실행

이 폴더의 핵심 실습만 연속 실행하려면:

```bash
python practice/10_diffusion_policy/00_run_core.py
```

전체 training을 자동으로 수행하는 명령이 아니라, 첫 바퀴에서 봐야 할 핵심 메커니즘만 실행한다.
생성된 그림은 `outputs/10_diffusion_policy/`에서 확인한다.

## 준비

```bash
bash scripts/setup_diffusion_policy.sh
bash scripts/download_pusht.sh
```

## 파일 / 실행

```bash
python practice/10_diffusion_policy/01_data.py
python practice/10_diffusion_policy/02_diffusion.py
python practice/10_diffusion_policy/03_policy.py
python practice/10_diffusion_policy/04_train.py --epochs 10
python practice/10_diffusion_policy/05_rollout.py
python practice/10_diffusion_policy/06_analyze.py
```

- `01_data.py`: 실제 Push-T image / agent_pos / 16-step action chunk
- `02_diffusion.py`: action forward diffusion과 epsilon target
- `03_policy.py`: official image-conditioned policy의 `compute_loss`
- `04_train.py`: official policy/data/loss 기반 scaled optimization
- `05_rollout.py`: 16-step prediction 중 다음 8 action 실행을 시각화
- `06_analyze.py`: 같은 observation에서 stochastic action samples 비교

실제 environment success-rate와 논문 수치 비교는 full checkpoint + official runner 단계에서 수행한다.
