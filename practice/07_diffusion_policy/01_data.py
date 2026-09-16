import matplotlib.pyplot as plt
import torch

from common import make_dataset, OUT, HORIZON, N_OBS_STEPS, N_ACTION_STEPS


ds = make_dataset()
sample = ds[0]
image = sample["obs"]["image"]
pos = sample["obs"]["agent_pos"]
action = sample["action"]

print("dataset samples:", len(ds))
print("image sequence:", image.shape, image.dtype, float(image.min()), float(image.max()))
print("agent_pos:", pos.shape, pos.dtype)
print("action:", action.shape, action.dtype)
print("paper horizons: prediction=", HORIZON, "obs=", N_OBS_STEPS, "execute=", N_ACTION_STEPS)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(image[0].permute(1, 2, 0).clamp(0, 1))
axes[0].set_title("Push-T obs t=0")
axes[1].imshow(image[1].permute(1, 2, 0).clamp(0, 1))
axes[1].set_title("Push-T obs t=1")
axes[2].plot(action[:, 0], action[:, 1], "o-")
axes[2].scatter(pos[:N_OBS_STEPS, 0], pos[:N_OBS_STEPS, 1], marker="x", s=80)
axes[2].set_title("16-step action chunk")
axes[2].set_aspect("equal", adjustable="box")
for ax in axes[:2]: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "01_pusht_data.png", dpi=150); plt.show()
