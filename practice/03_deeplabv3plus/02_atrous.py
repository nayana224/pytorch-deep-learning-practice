import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from pathlib import Path

OUT = Path("outputs/03_deeplabv3plus"); OUT.mkdir(parents=True, exist_ok=True)

x = torch.zeros(1, 1, 31, 31); x[0, 0, 15, 15] = 1
kernel = torch.ones(1, 1, 3, 3)
fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, rate in zip(axes, [1, 2, 4, 6]):
    y = F.conv2d(x, kernel, padding=rate, dilation=rate)
    ax.imshow(y[0, 0], cmap="viridis")
    ax.set_title(f"dilation={rate}")
    ax.axis("off")
plt.tight_layout(); plt.savefig(OUT / "02_atrous_receptive_field.png", dpi=150); plt.show()
