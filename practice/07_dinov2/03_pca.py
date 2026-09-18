"""DINOv2 patch feature가 이미지 안의 semantic structure를 갖는지 PCA로 본다.

DINOv2를 다시 pretrain하지 않는다.
공식 pretrained ViT의 patch token을 꺼내고 여러 이미지의 feature를
같은 PCA 공간에 투영해 비슷한 영역이 비슷한 색으로 나타나는지 관찰한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import OxfordIIITPet

from common import extract_features, image_transform, load_model


OUTPUT_DIR = Path("outputs/07_dinov2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model, device = load_model()
transform = image_transform()

try:
    dataset = OxfordIIITPet(
        "data/07_dinov2",
        split="test",
        download=False,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "Oxford-IIIT Pets가 없습니다. 먼저 실행하세요:\n"
        "  python scripts/download_torchvision_data.py pets"
    ) from error

images = []
patch_feature_sets = []

# 서로 다른 이미지의 patch feature를 같은 PCA basis에 투영해야
# 색 패턴을 이미지 간에 비교할 수 있다.
for index in [0, 1, 2]:
    image, _ = dataset[index]
    images.append(image)

    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        _, patch_tokens = extract_features(model, x)

    patch_feature_sets.append(patch_tokens[0].cpu())

all_patch_features = torch.cat(patch_feature_sets, dim=0)

# PCA 전에 전체 patch feature의 평균을 뺀다.
mean_feature = all_patch_features.mean(dim=0, keepdim=True)
centered = all_patch_features - mean_feature

# 첫 3개 principal component를 RGB처럼 사용한다.
_, _, principal_directions = torch.pca_lowrank(centered, q=3)
pca_rgb = centered @ principal_directions

minimum = pca_rgb.amin(dim=0)
maximum = pca_rgb.amax(dim=0)
pca_rgb = (pca_rgb - minimum) / (maximum - minimum + 1e-6)

fig, axes = plt.subplots(2, 3, figsize=(10, 7))
offset = 0

for column, (image, patch_features) in enumerate(
    zip(images, patch_feature_sets)
):
    num_patches = patch_features.shape[0]
    side = int(num_patches ** 0.5)

    axes[0, column].imshow(image)
    axes[0, column].set_title("input")

    patch_rgb = pca_rgb[offset : offset + num_patches]
    patch_rgb = patch_rgb.reshape(side, side, 3)

    axes[1, column].imshow(patch_rgb)
    axes[1, column].set_title("DINOv2 patch-feature PCA")

    axes[0, column].axis("off")
    axes[1, column].axis("off")

    offset += num_patches

fig.suptitle("label 없이 학습한 patch feature가 semantic region을 구분하는가?")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_patch_pca.png", dpi=150)
plt.show()
