from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet

from common import extract_features, image_transform, load_model


OUTPUT_DIR = Path("outputs/05_dinov2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model, device = load_model()
transform = image_transform()

try:
    dataset = OxfordIIITPet(
        "data/05_dinov2",
        split="test",
        download=False,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "Oxford-IIIT Pets is not prepared. Run: "
        "python scripts/download_torchvision_data.py pets"
    ) from error

features = []
num_images = min(100, len(dataset))

for index in range(num_images):
    image, _ = dataset[index]
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        class_feature, _ = extract_features(model, x)

    class_feature = F.normalize(class_feature, dim=-1)
    features.append(class_feature.cpu())

features = torch.cat(features, dim=0)

query_index = 0
query_feature = features[query_index]
similarity = features @ query_feature
nearest_indices = similarity.topk(6).indices.tolist()

fig, axes = plt.subplots(1, 6, figsize=(15, 3))

for ax, index in zip(axes, nearest_indices):
    image, label = dataset[index]
    class_name = dataset.classes[label]

    ax.imshow(image)
    ax.set_title(f"{class_name}\ncos={similarity[index]:.2f}")
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_nearest_neighbors.png", dpi=150)
plt.show()

print("Frozen CLS feature nearest-neighbor retrieval")
print("Inspect semantic matches and failure cases.")
