from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torchvision.datasets import VOCSegmentation

from data import DATA_DIR, pair_to_tensor
from deeplabv3plus import DeepLabV3Plus


OUTPUT_DIR = Path("outputs/03_deeplabv3plus")
CHECKPOINT = OUTPUT_DIR / "v3plus.pt"


def voc_colormap():
    cmap = np.zeros((21, 3), dtype=np.float32)

    for class_index in range(21):
        red = 0
        green = 0
        blue = 0
        value = class_index

        for bit in range(8):
            red |= ((value >> 0) & 1) << (7 - bit)
            green |= ((value >> 1) & 1) << (7 - bit)
            blue |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3

        cmap[class_index] = np.array([red, green, blue]) / 255.0

    return ListedColormap(cmap)


def prepare_segmentation_for_display(mask):
    mask = mask.detach().cpu().clone()
    ignore = mask == 255

    visible = mask.clone()
    visible[ignore] = 0

    return visible, ignore


def normalize_map(feature):
    feature = feature.detach().cpu()
    feature = feature - feature.min()
    return feature / feature.max().clamp_min(1e-8)


def mean_feature_map(feature):
    return normalize_map(feature[0].abs().mean(dim=0))


def semantic_boundary(target):
    boundary = torch.zeros_like(target, dtype=torch.bool)
    valid = target != 255

    boundary[1:] |= target[1:] != target[:-1]
    boundary[:-1] |= target[:-1] != target[1:]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, :-1] != target[:, 1:]

    return boundary & valid


if not CHECKPOINT.exists():
    raise SystemExit(
        "Run: python practice/03_deeplabv3plus/04_train.py "
        "--variant v3plus --epochs 20"
    )

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = torch.load(CHECKPOINT, map_location=device)
model = DeepLabV3Plus(
    num_classes=21,
    output_stride=16,
    use_decoder=True,
).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

try:
    raw_dataset = VOCSegmentation(
        DATA_DIR,
        year="2012",
        image_set="val",
        download=False,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "PASCAL VOC 2012 is not prepared. "
        "Run: bash scripts/download_voc2012.sh"
    ) from error

image, mask = raw_dataset[0]
image_tensor, target = pair_to_tensor(image, mask, train=False)

with torch.no_grad():
    logits, features = model(
        image_tensor.unsqueeze(0).to(device),
        return_features=True,
    )
    probabilities = F.softmax(logits, dim=1)

prediction = logits.argmax(dim=1)[0].cpu()
confidence = probabilities.max(dim=1).values[0].cpu()
error = (prediction != target) & (target != 255)
boundary = semantic_boundary(target)
boundary_error = error & boundary

gt_vis, gt_ignore = prepare_segmentation_for_display(target)
pred_vis, _ = prepare_segmentation_for_display(prediction)
segmentation_cmap = voc_colormap()

print("low-level feature :", features["low"].shape)
print("high-level feature:", features["high"].shape)
print("ASPP feature      :", features["aspp"].shape)
print("low-level -> 48   :", features["low48"].shape)
print("concat            :", features["concat"].shape)
print("decoded           :", features["decoded"].shape)
print(
    "boundary pixel accuracy:",
    1.0 - boundary_error.sum().item() / boundary.sum().clamp_min(1).item(),
)

# 1) Prediction visualization
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Input")

axes[0, 1].imshow(
    gt_vis,
    cmap=segmentation_cmap,
    vmin=0,
    vmax=20,
)
axes[0, 1].imshow(
    gt_ignore,
    cmap="gray",
    vmin=0,
    vmax=1,
    alpha=0.55,
)
axes[0, 1].set_title("GT (gray = ignore 255)")

axes[0, 2].imshow(
    pred_vis,
    cmap=segmentation_cmap,
    vmin=0,
    vmax=20,
)
axes[0, 2].set_title("Prediction")

axes[1, 0].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
axes[1, 0].set_title("Max class probability")

axes[1, 1].imshow(error, cmap="gray")
axes[1, 1].set_title("All pixel errors")

axes[1, 2].imshow(boundary_error, cmap="gray")
axes[1, 2].set_title("Errors on GT boundaries")

for ax in axes.ravel():
    ax.axis("off")

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_prediction_analysis.png", dpi=160)

# 2) ASPP branch visualization
branch_names = [
    "aspp_1x1",
    "aspp_rate6",
    "aspp_rate12",
    "aspp_rate18",
    "aspp_image_pool",
]

fig2, axes2 = plt.subplots(1, 5, figsize=(16, 3.5))

for ax, name in zip(axes2, branch_names):
    ax.imshow(mean_feature_map(features[name]), cmap="viridis")
    ax.set_title(name.replace("aspp_", ""))
    ax.axis("off")

fig2.suptitle("ASPP branches: different context scales")
fig2.tight_layout()
fig2.savefig(OUTPUT_DIR / "05_aspp_branches.png", dpi=160)

# 3) Low-level -> context -> decoder feature flow
flow_names = ["low", "high", "aspp", "low48", "concat", "decoded"]
fig3, axes3 = plt.subplots(2, 3, figsize=(12, 8))

for ax, name in zip(axes3.ravel(), flow_names):
    ax.imshow(mean_feature_map(features[name]), cmap="viridis")
    ax.set_title(f"{name}\n{tuple(features[name].shape)}")
    ax.axis("off")

fig3.suptitle("DeepLabv3+ feature flow")
fig3.tight_layout()
fig3.savefig(OUTPUT_DIR / "05_feature_flow.png", dpi=160)

plt.show()
