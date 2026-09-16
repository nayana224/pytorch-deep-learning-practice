from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR100


OUT = Path("outputs/04_vit")
OUT.mkdir(parents=True, exist_ok=True)

try:
    image, _ = CIFAR100(
        "data/04_vit",
        train=True,
        download=False,
    )[0]
except RuntimeError as error:
    raise FileNotFoundError(
        "CIFAR-100 is not prepared. Run: "
        "python scripts/download_torchvision_data.py cifar100"
    ) from error

x = transforms.ToTensor()(transforms.Resize((384, 384))(image))
patches = (
    x.unfold(1, 16, 16)
    .unfold(2, 16, 16)
    .permute(1, 2, 0, 3, 4)
    .reshape(-1, 3, 16, 16)
)

print("image:", x.shape)
print("patches:", patches.shape)
print("sequence length:", patches.shape[0])

fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for index, ax in enumerate(axes.flat):
    ax.imshow(patches[index].permute(1, 2, 0))
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "02_patches.png", dpi=150)
plt.show()
