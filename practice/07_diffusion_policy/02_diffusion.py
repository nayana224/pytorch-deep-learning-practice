import matplotlib.pyplot as plt
import torch

from common import make_dataset, make_scheduler, OUT


ds = make_dataset()
normalizer = ds.get_normalizer()
action = ds[0]["action"].unsqueeze(0)
naction = normalizer["action"].normalize(action)
scheduler = make_scheduler()
noise = torch.randn_like(naction)
steps = [0, 20, 50, 99]

fig, axes = plt.subplots(1, len(steps) + 1, figsize=(16, 3))
axes[0].plot(naction[0, :, 0], naction[0, :, 1], "o-")
axes[0].set_title("clean normalized action")
for ax, t in zip(axes[1:], steps):
    tt = torch.tensor([t], dtype=torch.long)
    noisy = scheduler.add_noise(naction, noise, tt)
    ax.plot(noisy[0, :, 0], noisy[0, :, 1], "o-")
    ax.set_title(f"forward diffusion t={t}")
for ax in axes: ax.set_aspect("equal", adjustable="box")
plt.tight_layout(); plt.savefig(OUT / "02_action_diffusion.png", dpi=150); plt.show()
print("clean:", naction.shape, "noise target:", noise.shape)
print("Training target for prediction_type='epsilon' is the sampled noise itself.")
