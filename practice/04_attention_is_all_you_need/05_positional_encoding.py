"""Transformer의 sinusoidal positional encoding을 직접 확인한다.

Self-Attention 자체에는 token 순서 정보가 없기 때문에,
논문은 각 위치마다 서로 다른 sin/cos 값을 더해 position을 알려준다.
"""

from pathlib import Path
import math

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/04_attention_is_all_you_need")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sequence_length = 16
d_model = 8

positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
dimensions = torch.arange(0, d_model, 2, dtype=torch.float32)

# 논문 식:
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
divisor = torch.exp(
    dimensions * (-math.log(10000.0) / d_model)
)

positional_encoding = torch.zeros(sequence_length, d_model)
positional_encoding[:, 0::2] = torch.sin(positions * divisor)
positional_encoding[:, 1::2] = torch.cos(positions * divisor)

print("positional encoding shape:", tuple(positional_encoding.shape))
print("position 0:", positional_encoding[0])
print("position 1:", positional_encoding[1])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

image = axes[0].imshow(
    positional_encoding.numpy(),
    aspect="auto",
)
axes[0].set_title("Sinusoidal Positional Encoding")
axes[0].set_xlabel("embedding dimension")
axes[0].set_ylabel("token position")
fig.colorbar(image, ax=axes[0], fraction=0.046)

# 몇 개 차원을 따로 그리면 위치에 따라 서로 다른 주기의 sin/cos가 변하는 것을 볼 수 있다.
for dimension in range(min(4, d_model)):
    axes[1].plot(
        range(sequence_length),
        positional_encoding[:, dimension],
        marker="o",
        label=f"dim {dimension}",
    )

axes[1].set_title("Encoding values across token positions")
axes[1].set_xlabel("token position")
axes[1].set_ylabel("encoding value")
axes[1].legend()
axes[1].grid(True)

fig.suptitle("Positional encoding injects token-order information")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "05_positional_encoding.png", dpi=160)
plt.show()
