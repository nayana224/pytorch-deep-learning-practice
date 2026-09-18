"""U-Net의 핵심 구조를 Figure 1 순서대로 확인한다.

첫 바퀴의 목적은 전체 학습이 아니라 다음 세 가지를 눈으로 확인하는 것이다.
1. valid convolution 때문에 spatial size가 줄어든다.
2. encoder의 고해상도 feature를 center crop한다.
3. crop한 encoder feature와 upsample된 decoder feature를 channel 방향으로 concat한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from unet import UNet, print_feature_shapes


OUTPUT_DIR = Path("outputs/02_unet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    """feature map을 그림으로 보기 쉽도록 0~1 범위로 정규화한다."""
    x = x.detach().cpu()
    x = x - x.min()
    return x / x.max().clamp_min(1e-8)


def main() -> None:
    torch.manual_seed(0)

    model = UNet(in_channels=1, num_classes=2)
    model.eval()

    # 원 논문 Figure 1은 572x572 입력 tile을 사용한다.
    # 실제 데이터가 아니라 구조와 shape 흐름만 확인하는 입력이다.
    x = torch.randn(1, 1, 572, 572)

    with torch.no_grad():
        logits, features = model(x, return_features=True)

    print("=== U-Net Figure 1 shape 흐름 ===")
    print_feature_shapes(features)
    print()
    print("최종 output shape:", logits.shape)
    print("논문 Figure 1의 기대 spatial size: 388 x 388")

    assert logits.shape == (1, 2, 388, 388)

    # encoder → bottleneck → decoder의 대표 지점만 뽑아 본다.
    # 모든 channel을 따로 그리기보다 채널별 절댓값 평균을 사용해
    # 각 단계가 어떤 spatial pattern을 유지하는지 빠르게 확인한다.
    names = [
        "enc1",
        "enc4",
        "bottleneck",
        "crop4",
        "up4",
        "concat4",
        "dec1",
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for ax, name in zip(axes, names):
        feature = features[name][0].abs().mean(dim=0)
        ax.imshow(normalize_map(feature), cmap="viridis")
        ax.set_title(f"{name}\n{tuple(features[name].shape)}")
        ax.axis("off")

    axes[-1].axis("off")
    fig.suptitle("U-Net: encoder → bottleneck → crop/copy → decoder")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_model_feature_overview.png", dpi=160)

    # U-Net에서 가장 중요한 skip connection을 따로 크게 본다.
    # crop4와 up4는 spatial size가 같아야 concat할 수 있다.
    crop4 = features["crop4"][0].abs().mean(dim=0)
    up4 = features["up4"][0].abs().mean(dim=0)
    concat4 = features["concat4"][0].abs().mean(dim=0)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    for ax, feature, title in zip(
        axes2,
        [crop4, up4, concat4],
        [
            "cropped encoder feature",
            "up-convolved decoder feature",
            "after channel concat",
        ],
    ):
        ax.imshow(normalize_map(feature), cmap="magma")
        ax.set_title(title)
        ax.axis("off")

    fig2.suptitle("U-Net skip connection: high-resolution encoder feature to decoder")
    fig2.tight_layout()
    fig2.savefig(OUTPUT_DIR / "02_model_crop_concat.png", dpi=160)

    plt.show()


if __name__ == "__main__":
    main()
