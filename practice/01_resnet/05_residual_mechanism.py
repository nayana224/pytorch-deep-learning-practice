"""ResNet의 핵심 식 F(x) + x를 feature map으로 직접 확인한다.

학습 성능을 보는 코드가 아니라 residual block 내부의 데이터 흐름을
눈으로 확인하기 위한 Level 2 mechanism visualization이다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

from resnet import ResidualBlock


OUTPUT_DIR = Path("outputs/01_resnet")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_map(x: torch.Tensor) -> torch.Tensor:
    """채널 평균 activation을 0~1 범위로 정규화한다."""
    x = x.detach().cpu()
    x = x - x.min()
    return x / x.max().clamp_min(1e-8)


torch.manual_seed(0)

block = ResidualBlock(16, 16)
block.eval()

# 실제 feature tensor와 같은 [B, C, H, W] 형태의 입력을 사용한다.
x = torch.randn(1, 16, 32, 32)

with torch.no_grad():
    out, residual, identity = block(x, return_parts=True)

# 각 tensor의 채널 방향 평균 절댓값을 2D map으로 만든다.
x_map = normalize_map(x[0].abs().mean(dim=0))
residual_map = normalize_map(residual[0].abs().mean(dim=0))
identity_map = normalize_map(identity[0].abs().mean(dim=0))
sum_before_relu = residual + identity
sum_map = normalize_map(sum_before_relu[0].abs().mean(dim=0))
out_map = normalize_map(out[0].abs().mean(dim=0))

print("x shape:", tuple(x.shape))
print("F(x) shape:", tuple(residual.shape))
print("shortcut x shape:", tuple(identity.shape))
print("F(x)+x shape:", tuple(sum_before_relu.shape))
print("block output shape:", tuple(out.shape))

fig, axes = plt.subplots(1, 5, figsize=(17, 3.5))

items = [
    (x_map, "input x"),
    (residual_map, "residual F(x)"),
    (identity_map, "shortcut x"),
    (sum_map, "F(x) + x"),
    (out_map, "ReLU(F(x)+x)"),
]

for ax, (feature, title) in zip(axes, items):
    ax.imshow(feature, cmap="viridis")
    ax.set_title(title)
    ax.axis("off")

fig.suptitle("ResNet residual block 내부: F(x)와 shortcut이 실제로 더해진다")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_residual_mechanism.png", dpi=160)
plt.show()
