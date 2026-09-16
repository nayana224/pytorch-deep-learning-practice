import matplotlib.pyplot as plt
import torch

from common import OUT, make_dataset, make_scheduler


# 1. Clean demonstration action sequence
train_dataset = make_dataset()
normalizer = train_dataset.get_normalizer()

action = train_dataset[0]["action"].unsqueeze(0)
normalized_action = normalizer["action"].normalize(action)

# 2. DDPM forward process
scheduler = make_scheduler()
noise = torch.randn_like(normalized_action)

timesteps = [0, 20, 50, 99]

fig, axes = plt.subplots(1, len(timesteps) + 1, figsize=(16, 3))

axes[0].plot(
    normalized_action[0, :, 0],
    normalized_action[0, :, 1],
    "o-",
)
axes[0].set_title("clean action")

for ax, timestep in zip(axes[1:], timesteps):
    t = torch.tensor([timestep], dtype=torch.long)
    noisy_action = scheduler.add_noise(
        normalized_action,
        noise,
        t,
    )

    ax.plot(
        noisy_action[0, :, 0],
        noisy_action[0, :, 1],
        "o-",
    )
    ax.set_title(f"t={timestep}")

for ax in axes:
    ax.set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.savefig(OUT / "02_action_diffusion.png", dpi=150)
plt.show()

print("clean action:", normalized_action.shape)
print("noise target:", noise.shape)
print("epsilon prediction target = sampled Gaussian noise")
