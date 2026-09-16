from pathlib import Path
import sys

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

ROOT = Path(__file__).resolve().parents[2]
OFFICIAL = ROOT / "external" / "diffusion_policy"
DATA_ROOT = ROOT / "data" / "07_diffusion_policy"
OUT = ROOT / "outputs" / "07_diffusion_policy"
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 16
N_OBS_STEPS = 2
N_ACTION_STEPS = 8
SHAPE_META = {
    "obs": {
        "image": {"shape": [3, 96, 96], "type": "rgb"},
        "agent_pos": {"shape": [2], "type": "low_dim"},
    },
    "action": {"shape": [2]},
}


def require_official_repo():
    if not OFFICIAL.exists():
        raise FileNotFoundError("Run: bash scripts/setup_diffusion_policy.sh")
    if str(OFFICIAL) not in sys.path:
        sys.path.insert(0, str(OFFICIAL))


def find_zarr():
    candidates = list(DATA_ROOT.rglob("pusht_cchi_v7_replay.zarr"))
    if not candidates:
        candidates = list(DATA_ROOT.rglob("*.zarr"))
    if not candidates:
        raise FileNotFoundError("Run: bash scripts/download_pusht.sh")
    return candidates[0]


def make_scheduler():
    return DDPMScheduler(
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        prediction_type="epsilon",
    )


def make_dataset():
    require_official_repo()
    from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
    return PushTImageDataset(
        zarr_path=str(find_zarr()),
        horizon=HORIZON,
        pad_before=N_OBS_STEPS - 1,
        pad_after=N_ACTION_STEPS - 1,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )


def make_policy(device=None):
    require_official_repo()
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=make_scheduler(),
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        n_obs_steps=N_OBS_STEPS,
        num_inference_steps=100,
        obs_as_global_cond=True,
        crop_shape=(84, 84),
        diffusion_step_embed_dim=128,
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        obs_encoder_group_norm=True,
        eval_fixed_crop=True,
    ).to(device)
    return policy, device
