from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch


DATA_DIR = Path("data/02_unet/isbi2012")
IMAGE_PATH = DATA_DIR / "train-volume.tif"
LABEL_PATH = DATA_DIR / "train-labels.tif"
OUTPUT_DIR = Path("outputs/02_unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    images = tifffile.imread(IMAGE_PATH)
    labels = tifffile.imread(LABEL_PATH)

    print("=== raw TIFF stack ===")
    print("images shape:", images.shape)
    print("images dtype:", images.dtype)
    print("images min/max:", images.min(), images.max())
    print()
    print("labels shape:", labels.shape)
    print("labels dtype:", labels.dtype)
    print("labels min/max:", labels.min(), labels.max())
    print("labels unique:", np.unique(labels))

    index = 0
    image = images[index]
    label = labels[index]

    membrane_ratio = np.mean(label == 0)
    interior_ratio = np.mean(label == 255)

    print()
    print(f"slice {index} membrane(0) ratio: {membrane_ratio:.4f}")
    print(f"slice {index} interior(255) ratio: {interior_ratio:.4f}")

    # Explicit conversion so the transformation is visible while studying.
    image_tensor = torch.from_numpy(image.copy()).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # [H,W] -> [C,H,W]

    label_tensor = torch.from_numpy(label.copy())
    label_tensor = (label_tensor == 255).long()  # 0=membrane, 1=cell interior

    print()
    print("=== one sample as PyTorch tensors ===")
    print("image tensor shape:", image_tensor.shape)
    print("image tensor dtype:", image_tensor.dtype)
    print("image tensor min/max:", image_tensor.min().item(), image_tensor.max().item())
    print("label tensor shape:", label_tensor.shape)
    print("label tensor dtype:", label_tensor.dtype)
    print("label tensor unique:", torch.unique(label_tensor))

    # Visualization 1: raw image, GT, membrane overlay
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("EM image")
    axes[0].axis("off")

    axes[1].imshow(label, cmap="gray", vmin=0, vmax=255)
    axes[1].set_title("GT: black=membrane, white=cell interior")
    axes[1].axis("off")

    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(label == 0, alpha=0.45, cmap="Reds")
    axes[2].set_title("Membrane overlay: label == 0")
    axes[2].axis("off")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_data_input_gt_overlay.png", dpi=160)

    # Visualization 2: compare the two binary classes directly.
    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 5))

    axes2[0].imshow(image, cmap="gray")
    axes2[0].imshow(label == 0, alpha=0.4, cmap="Reds")
    axes2[0].set_title("class 0: membrane")
    axes2[0].axis("off")

    axes2[1].imshow(image, cmap="gray")
    axes2[1].imshow(label == 255, alpha=0.35, cmap="Blues")
    axes2[1].set_title("class 1: cell interior")
    axes2[1].axis("off")

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "01_data_class_overlay.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()
