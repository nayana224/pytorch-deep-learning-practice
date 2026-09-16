from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import VOCSegmentation

from data import DATA_DIR, pair_to_tensor


OUTPUT_DIR = Path("outputs/03_deeplabv3plus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    raw_dataset = VOCSegmentation(
        DATA_DIR,
        year="2012",
        image_set="train",
        download=False,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "PASCAL VOC 2012 is not prepared. Run: bash scripts/download_voc2012.sh"
    ) from error

image, mask = raw_dataset[0]
image_tensor, mask_tensor = pair_to_tensor(image, mask, train=False)

print("dataset size :", len(raw_dataset))
print("image tensor :", image_tensor.shape, image_tensor.dtype)
print("mask tensor  :", mask_tensor.shape, mask_tensor.dtype)
print("mask classes :", torch.unique(mask_tensor))
print("255 means ignore label")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title("PASCAL VOC 2012 image")
axes[1].imshow(mask, cmap="tab20")
axes[1].set_title("semantic segmentation GT")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_voc.png", dpi=150)
plt.show()
