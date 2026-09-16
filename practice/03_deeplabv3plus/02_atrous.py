from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


OUTPUT_DIR = Path("outputs/03_deeplabv3plus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Single active pixel so the sampling pattern is easy to see.
x = torch.zeros(1, 1, 31, 31)
x[0, 0, 15, 15] = 1.0

kernel = torch.ones(1, 1, 3, 3)
dilation_rates = [1, 2, 4, 6]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

for ax, rate in zip(axes, dilation_rates):
    y = F.conv2d(
        x,
        kernel,
        padding=rate,
        dilation=rate,
    )

    ax.imshow(y[0, 0], cmap="viridis")
    ax.set_title(f"dilation={rate}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_atrous_receptive_field.png", dpi=150)
plt.show()
