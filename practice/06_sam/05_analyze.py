import matplotlib.pyplot as plt
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from sam_utils import CHECKPOINT, OUT, load_sample


image, gt_mask, annotation, image_path = load_sample()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
sam = sam.to(device)

# The paper's fully automatic data-engine stage uses a regular grid
# of foreground-point prompts. Here we inspect the same idea through
# the official automatic-mask generator.
generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=16,
)

masks = generator.generate(image)
masks = sorted(masks, key=lambda mask: mask["area"], reverse=True)

print("generated masks:", len(masks))
print("largest areas:", [mask["area"] for mask in masks[:10]])

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(image)

for mask in masks[:20]:
    ax.contour(
        mask["segmentation"],
        levels=[0.5],
        linewidths=0.7,
    )

ax.set_title("automatic mask generation: top 20 by area")
ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "05_auto_masks.png", dpi=150)
plt.show()
