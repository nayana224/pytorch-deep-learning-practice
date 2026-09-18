"""Vision Transformer 실습 파일.

논문 구조와 핵심 메커니즘을 읽기 쉽게 따라가기 위한 공부용 코드다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR100


DATA = Path("data/05_vit")
OUT = Path("outputs/05_vit")
OUT.mkdir(parents=True, exist_ok=True)

try:
    raw = CIFAR100(DATA, train=True, download=False)
except RuntimeError as error:
    raise FileNotFoundError(
        "CIFAR-100 is not prepared. Run: "
        "python scripts/download_torchvision_data.py cifar100"
    ) from error

image, label = raw[0]
print(
    "CIFAR-100 train:", len(raw),
    "raw size:", image.size,
    "label:", label, raw.classes[label],
)

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(image)
axes[0].set_title("original 32x32")

resized = transforms.Resize((384, 384))(image)
axes[1].imshow(resized)
axes[1].set_title("paper fine-tune resolution 384")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "01_cifar100.png", dpi=150)
plt.show()
