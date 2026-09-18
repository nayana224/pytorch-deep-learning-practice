"""DeepLabv3+ 논문 실습 코드.

Atrous convolution, ASPP, low-level feature와 decoder가
semantic segmentation에 어떻게 사용되는지 확인하기 위한 공부용 코드다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from torchvision import transforms
from torchvision.datasets import VOCSegmentation


ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_REPO = ROOT / "external" / "DeepLabV3Plus-Pytorch"
CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "deeplabv3plus"
    / "best_deeplabv3plus_resnet101_voc_os16.pth"
)
OUTPUT_DIR = ROOT / "outputs" / "03_deeplabv3plus"
DATA_DIR = ROOT / "data" / "03_deeplabv3plus"

if not EXTERNAL_REPO.exists() or not CHECKPOINT.exists():
    raise SystemExit(
        "Run first:\n"
        "  bash scripts/setup_deeplabv3plus_pretrained.sh"
    )

sys.path.insert(0, str(EXTERNAL_REPO))

import network  # noqa: E402


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


def mask_to_tensor(mask):
    return torch.from_numpy(np.array(mask, dtype=np.int64))


def prepare_segmentation_for_display(mask):
    mask = mask.detach().cpu().clone()
    ignore = mask == 255
    visible = mask.clone()
    visible[ignore] = 0
    return visible, ignore


def semantic_boundary(target):
    boundary = torch.zeros_like(target, dtype=torch.bool)
    valid = target != 255

    boundary[1:] |= target[1:] != target[:-1]
    boundary[:-1] |= target[:-1] != target[1:]
    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, :-1] != target[:, 1:]

    return boundary & valid


try:
    dataset = VOCSegmentation(
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = network.modeling.deeplabv3plus_resnet101(
    num_classes=21,
    output_stride=16,
)

checkpoint = torch.load(CHECKPOINT, map_location="cpu")
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

sample_indices = [0, 1, 2]
segmentation_cmap = voc_colormap()

fig, axes = plt.subplots(
    len(sample_indices),
    6,
    figsize=(18, 4.5 * len(sample_indices)),
)

if len(sample_indices) == 1:
    axes = axes[None, :]

for row, sample_index in enumerate(sample_indices):
    image, mask = dataset[sample_index]
    x = transform(image).unsqueeze(0).to(device)
    target = mask_to_tensor(mask)

    with torch.no_grad():
        logits = model(x)
        logits = F.interpolate(
            logits,
            size=target.shape,
            mode="bilinear",
            align_corners=False,
        )
        probabilities = F.softmax(logits, dim=1)

    prediction = logits.argmax(dim=1)[0].cpu()
    confidence = probabilities.max(dim=1).values[0].cpu()

    valid = target != 255
    error = (prediction != target) & valid
    boundary = semantic_boundary(target)
    boundary_error = error & boundary

    gt_vis, gt_ignore = prepare_segmentation_for_display(target)
    pred_vis, _ = prepare_segmentation_for_display(prediction)

    axes[row, 0].imshow(image)
    axes[row, 0].set_title("Input")

    axes[row, 1].imshow(
        gt_vis,
        cmap=segmentation_cmap,
        vmin=0,
        vmax=20,
    )
    axes[row, 1].imshow(
        gt_ignore,
        cmap="gray",
        vmin=0,
        vmax=1,
        alpha=0.55,
    )
    axes[row, 1].set_title("GT")

    axes[row, 2].imshow(
        pred_vis,
        cmap=segmentation_cmap,
        vmin=0,
        vmax=20,
    )
    axes[row, 2].set_title("Pretrained prediction")

    axes[row, 3].imshow(confidence, cmap="viridis", vmin=0, vmax=1)
    axes[row, 3].set_title("Confidence")

    axes[row, 4].imshow(error, cmap="gray")
    axes[row, 4].set_title("Error map")

    axes[row, 5].imshow(boundary_error, cmap="gray")
    axes[row, 5].set_title("Boundary error")

    for col in range(6):
        axes[row, col].axis("off")

    pixel_accuracy = (
        ((prediction == target) & valid).sum().item()
        / valid.sum().clamp_min(1).item()
    )
    boundary_accuracy = (
        1.0
        - boundary_error.sum().item()
        / boundary.sum().clamp_min(1).item()
    )

    print(
        f"sample={sample_index} "
        f"pixel_acc={pixel_accuracy:.4f} "
        f"boundary_acc={boundary_accuracy:.4f}"
    )

fig.suptitle(
    "Pretrained DeepLabv3+ ResNet101 on PASCAL VOC 2012",
    fontsize=14,
)
fig.tight_layout()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(
    OUTPUT_DIR / "07_pretrained_prediction_analysis.png",
    dpi=160,
)
plt.show()
