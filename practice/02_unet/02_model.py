"""U-Net 논문 실습 코드.

Contracting path, valid convolution, crop-and-copy skip connection,
expanding path와 segmentation 결과를 확인하기 위한 공부용 코드다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from unet import UNet, print_feature_shapes


OUTPUT_DIR = Path("outputs/02_unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().cpu()
    x = x - x.min()
    denom = x.max().clamp_min(1e-8)
    return x / denom


def main() -> None:
    torch.manual_seed(0)

    model = UNet(in_channels=1, num_classes=2)
    model.eval()

    # Figure 1 in the original paper uses a 572x572 input tile.
    x = torch.randn(1, 1, 572, 572)

    with torch.no_grad():
        logits, features = model(x, return_features=True)

    print("=== Original U-Net Figure 1 shape trace ===")
    print_feature_shapes(features)
    print()
    print("final output shape:", logits.shape)
    print("expected Figure 1 output spatial size: 388 x 388")

    assert logits.shape == (1, 2, 388, 388)

    # Visualize mean absolute activation at important points.
    names = ["enc1", "enc4", "bottleneck", "crop4", "up4", "concat4", "dec1"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        feature = features[name][0].abs().mean(dim=0)
        ax.imshow(normalize_map(feature), cmap="viridis")
        ax.set_title(f"{name}\n{tuple(features[name].shape)}")
        ax.axis("off")

    axes[-1].axis("off")
    fig.suptitle("U-Net feature-map overview (random input; structure check)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_model_feature_overview.png", dpi=160)

    # Crop+concat visualization: compare spatial alignment explicitly.
    crop4 = features["crop4"][0].abs().mean(dim=0)
    up4 = features["up4"][0].abs().mean(dim=0)
    concat4 = features["concat4"][0].abs().mean(dim=0)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, feature, title in zip(
        axes2,
        [crop4, up4, concat4],
        ["cropped encoder feature", "up-convolved decoder feature", "after channel concat"],
    ):
        ax.imshow(normalize_map(feature), cmap="magma")
        ax.set_title(title)
        ax.axis("off")

    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "02_model_crop_concat.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()
