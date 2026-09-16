from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import OxfordIIITPet

from common import transform

OUT = Path("outputs/05_dinov2")
OUT.mkdir(parents=True, exist_ok=True)

ds = OxfordIIITPet("data/05_dinov2", split="test", target_types="category", download=True)
image, label = ds[0]
x = transform()(image)

print(
    "Oxford-IIIT Pets test:", len(ds),
    "image:", image.size,
    "tensor:", x.shape, x.dtype,
    "label:", label, ds.classes[label],
)

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])
vis = (x.permute(1, 2, 0) * std + mean).clamp(0, 1)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(image)
axes[0].set_title("paper benchmark image")
axes[1].imshow(vis)
axes[1].set_title("224x224 model input")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUT / "01_pets.png", dpi=150)
plt.show()
