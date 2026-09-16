from pathlib import Path
import sys

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = ROOT / "external" / "diffusion_policy"
DATA_DIR = ROOT / "data" / "07_diffusion_policy"
OUTPUT_DIR = ROOT / "outputs" / "07_diffusion_policy"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Official Push-T image-policy setting used in this practice.
PREDICTION_HORIZON = 16
OBSERVATION_HORIZON = 2
ACTION_HORIZON = 8

SHAPE_META = {
    "obs": {
        "image": {
            "shape": [3, 96, 96],
            "type": "rgb",
        },
        "agent_pos": {
            "shape": [2],
            "type": "low_dim",
        },
    },
    "action": {
        "shape": [2],
    },
}

# Existing script compatibility
OUT = OUTPUT_DIR
HORIZON = PREDICTION_HORIZON
N_OBS_STEPS = OBSERVATION_HORIZON
N_ACTION_STEPS = ACTION_HORIZON


def require_official_repo():
    if not OFFICIAL_REPO.exists():
        raise FileNotFoundError("Run bash scripts/setup_diffusion_policy.sh")

    if str(OFFICIAL_REPO) not in sys.path:
        sys.path.insert(0, str(OFFICIAL_REPO))


def find_push_t_data():
    exact = list(DATA_DIR.rglob("pusht_cchi_v7_replay.zarr"))
    if exact:
        return exact[0]

    any_zarr = list(DATA_DIR.rglob("*.zarr"))
    if any_zarr:
        return any_zarr[0]

    raise FileNotFoundError("Run bash scripts/download_pusht.sh")


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
        zarr_path=str(find_push_t_data()),
        horizon=PREDICTION_HORIZON,
        pad_before=OBSERVATION_HORIZON - 1,
        pad_after=ACTION_HORIZON - 1,
        seed=42,
        val_ratio=0.02,
        max_train_episodes=90,
    )


def make_policy():
    require_official_repo()

    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = DiffusionUnetHybridImagePolicy(
        shape_meta=SHAPE_META,
        noise_scheduler=make_scheduler(),
        horizon=PREDICTION_HORIZON,
        n_obs_steps=OBSERVATION_HORIZON,
        n_action_steps=ACTION_HORIZON,
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
    )

    policy = policy.to(device)
    return policy, device
