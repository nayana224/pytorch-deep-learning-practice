"""U-Net 논문 실습 코드.

Contracting path, valid convolution, crop-and-copy skip connection,
expanding path와 segmentation 결과를 확인하기 위한 공부용 코드다.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import tifffile
import torch
import torch.nn.functional as F

from unet import UNet, center_crop_target


DATA_DIR = Path("data/02_unet/isbi2012")
IMAGE_PATH = DATA_DIR / "train-volume.tif"
LABEL_PATH = DATA_DIR / "train-labels.tif"
OUTPUT_DIR = Path("outputs/02_unet")
CHECKPOINT_PATH = OUTPUT_DIR / "unet_baseline.pt"


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    x = x - x.min()
    return x / x.max().clamp_min(1e-8)


def membrane_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_membrane = pred == 0
    target_membrane = target == 0
    intersection = (pred_membrane & target_membrane).sum().item()
    union = (pred_membrane | target_membrane).sum().item()
    return intersection / union if union > 0 else 1.0


def membrane_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_membrane = pred == 0
    target_membrane = target == 0
    intersection = (pred_membrane & target_membrane).sum().item()
    denom = pred_membrane.sum().item() + target_membrane.sum().item()
    return 2.0 * intersection / denom if denom > 0 else 1.0


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"checkpoint not found: {CHECKPOINT_PATH}\n"
            "Run: python practice/02_unet/03_train.py --epochs 10"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    images = tifffile.imread(IMAGE_PATH)
    labels = tifffile.imread(LABEL_PATH)

    # First validation slice from the simple 24/6 split used in 03_train.py.
    index = 24
    image = torch.from_numpy(images[index].copy()).float() / 255.0
    image = image.unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]

    target = torch.from_numpy(labels[index].copy())
    target = (target == 255).long().unsqueeze(0).to(device)  # [1,H,W]

    model = UNet(in_channels=1, num_classes=2).to(device)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits, features = model(image, return_features=True)
        aligned_target = center_crop_target(target, logits)
        probabilities = F.softmax(logits, dim=1)
        prediction = logits.argmax(dim=1)

    membrane_probability = probabilities[0, 0].cpu()
    pred = prediction[0].cpu()
    gt = aligned_target[0].cpu()

    iou = membrane_iou(pred, gt)
    dice = membrane_dice(pred, gt)

    print("checkpoint epoch:", checkpoint.get("epoch"))
    print("model input shape:", tuple(image.shape))
    print("logits shape:", tuple(logits.shape))
    print("aligned GT shape:", tuple(aligned_target.shape))
    print(f"membrane IoU: {iou:.4f}")
    print(f"membrane Dice: {dice:.4f}")

    # Crop input to the output field-of-view for a fair visual comparison.
    input_gray = image[0, 0].cpu()
    out_h, out_w = pred.shape
    in_h, in_w = input_gray.shape
    top = (in_h - out_h) // 2
    left = (in_w - out_w) // 2
    input_crop = input_gray[top : top + out_h, left : left + out_w]

    error = pred != gt

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))

    axes[0, 0].imshow(input_crop, cmap="gray")
    axes[0, 0].set_title("Input (output FOV)")

    axes[0, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("GT: 0 membrane / 1 interior")

    axes[0, 2].imshow(membrane_probability, cmap="magma", vmin=0, vmax=1)
    axes[0, 2].set_title("P(membrane)")

    axes[1, 0].imshow(pred, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Prediction")

    axes[1, 1].imshow(input_crop, cmap="gray")
    axes[1, 1].imshow(pred == 0, cmap="Reds", alpha=0.4)
    axes[1, 1].set_title("Predicted membrane overlay")

    axes[1, 2].imshow(error, cmap="Reds")
    axes[1, 2].set_title(f"Error map\nIoU={iou:.3f}, Dice={dice:.3f}")

    for ax in axes.ravel():
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_prediction_analysis.png", dpi=160)

    # Feature visualization: average absolute activation across channels.
    feature_names = ["enc1", "enc3", "enc4", "bottleneck", "up4", "crop4", "concat4", "dec1"]
    fig2, axes2 = plt.subplots(2, 4, figsize=(15, 8))

    for ax, name in zip(axes2.ravel(), feature_names):
        fmap = features[name][0].abs().mean(dim=0)
        ax.imshow(normalize_map(fmap), cmap="viridis")
        ax.set_title(f"{name}\n{tuple(features[name].shape)}")
        ax.axis("off")

    fig2.suptitle("Encoder / skip / decoder feature overview")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "04_feature_analysis.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()
