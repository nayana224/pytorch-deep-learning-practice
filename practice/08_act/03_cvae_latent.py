"""ACT의 CVAE latent variable 흐름을 최소 tensor 예제로 확인한다.

실제 ACT 학습을 재현하지 않고,
demonstration action sequence -> latent distribution -> z sampling
이라는 데이터 흐름만 확인한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/08_act")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(3)

batch_size = 4
latent_dim = 2

# encoder가 demonstration으로부터 만든 posterior parameter라고 가정한다.
mu = torch.tensor(
    [
        [-1.0, 0.2],
        [-0.3, 0.8],
        [0.5, -0.5],
        [1.0, 0.4],
    ]
)
logvar = torch.tensor(
    [
        [-0.4, -0.2],
        [-0.2, -0.5],
        [-0.3, -0.3],
        [-0.5, -0.2],
    ]
)

# reparameterization trick: z = mu + sigma * epsilon
epsilon = torch.randn(batch_size, latent_dim)
std = torch.exp(0.5 * logvar)
z = mu + std * epsilon

print("mu shape:", tuple(mu.shape))
print("logvar shape:", tuple(logvar.shape))
print("sampled z shape:", tuple(z.shape))

fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(mu[:, 0], mu[:, 1], s=140, label="posterior mean mu")
ax.scatter(z[:, 0], z[:, 1], s=140, marker="x", label="sampled latent z")

for index in range(batch_size):
    ax.plot(
        [mu[index, 0], z[index, 0]],
        [mu[index, 1], z[index, 1]],
        linestyle="--",
    )

ax.set_title("ACT CVAE: sample latent z from the posterior")
ax.set_xlabel("z dim 0")
ax.set_ylabel("z dim 1")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_cvae_latent.png", dpi=160)
plt.show()
