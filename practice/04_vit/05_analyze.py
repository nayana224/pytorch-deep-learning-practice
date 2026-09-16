from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import CIFAR100

from vit import vit_b16, vit_tiny16


output_dir = Path("outputs/04_vit")
checkpoint = torch.load(output_dir / "tiny.pt", map_location="cpu")

image_size = checkpoint["size"]

if checkpoint["variant"] == "tiny":
    model = vit_tiny16(100, image_size)
else:
    model = vit_b16(100, image_size)

model.load_state_dict(checkpoint["model"])
model.eval()

dataset = CIFAR100(
    "data/04_vit",
    train=False,
    download=True,
)

image, label = dataset[0]

transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ]
)

x = transform(image).unsqueeze(0)

with torch.no_grad():
    logits, features = model(
        x,
        return_attention=True,
    )

# Last encoder block: average all heads, then CLS -> patch attention.
last_attention = features["attentions"][-1]
cls_attention = last_attention[0].mean(dim=0)[0, 1:]

patch_grid = image_size // 16
cls_attention = cls_attention.reshape(patch_grid, patch_grid)

attention_map = F.interpolate(
    cls_attention[None, None],
    size=(image_size, image_size),
    mode="bilinear",
    align_corners=False,
)[0, 0]

prediction = logits.argmax(dim=1).item()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

axes[0].imshow(image.resize((image_size, image_size)))
axes[0].set_title(
    f"GT={dataset.classes[label]}\n"
    f"Pred={dataset.classes[prediction]}"
)

axes[1].imshow(image.resize((image_size, image_size)))
axes[1].imshow(attention_map, alpha=0.55)
axes[1].set_title("last-layer CLS attention")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(output_dir / "05_attention.png", dpi=150)
plt.show()
