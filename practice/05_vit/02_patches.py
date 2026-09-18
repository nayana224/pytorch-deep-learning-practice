"""이미지가 ViT의 patch token sequence로 바뀌는 과정을 확인한다.

ViT의 첫 번째 핵심은 이미지를 CNN feature map으로 처리하는 대신,
고정 크기 patch로 잘라 sequence처럼 Transformer에 넣는 것이다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import CIFAR100


OUTPUT_DIR = Path("outputs/05_vit")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    image, _ = CIFAR100(
        "data/05_vit",
        train=True,
        download=False,
    )[0]
except RuntimeError as error:
    raise FileNotFoundError(
        "CIFAR-100이 없습니다. 먼저 실행하세요:\n"
        "  python scripts/download_torchvision_data.py cifar100"
    ) from error

# 원본 CIFAR-100은 32x32이므로 ViT-B/16 구조를 보기 위해
# 384x384로 resize한다. 이 실습의 목적은 성능이 아니라 patch 흐름이다.
x = transforms.ToTensor()(
    transforms.Resize((384, 384))(image)
)

# [C,H,W] → 16x16 non-overlapping patch들의 sequence로 변환한다.
# 384 / 16 = 24이므로 총 24x24 = 576개의 patch가 생긴다.
patches = (
    x.unfold(1, 16, 16)
    .unfold(2, 16, 16)
    .permute(1, 2, 0, 3, 4)
    .reshape(-1, 3, 16, 16)
)

print("image shape:", tuple(x.shape))
print("patch tensor shape:", tuple(patches.shape))
print("patch sequence length:", patches.shape[0])

# 전체 576개를 다 그리기보다 앞의 32개만 확인한다.
fig, axes = plt.subplots(4, 8, figsize=(10, 5))
for index, ax in enumerate(axes.flat):
    ax.imshow(patches[index].permute(1, 2, 0))
    ax.set_title(f"patch {index}", fontsize=7)
    ax.axis("off")

fig.suptitle("ViT: image → 16x16 patch sequence")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_patches.png", dpi=150)
plt.show()
