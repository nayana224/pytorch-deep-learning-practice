"""Scaled Dot-Product Attention을 수식 그대로 계산한다."""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/04_attention_is_all_you_need")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokens = ["I", "study", "deep", "learning"]

# 이해를 위해 Q와 K를 직접 정의한다.
# 행 하나가 하나의 token에 대응한다.
q = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0, 1.0],
    ]
)

k = torch.tensor(
    [
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.5, 0.0, 1.0],
        [0.5, 0.5, 1.0, 0.0],
        [0.0, 1.0, 1.5, 1.0],
    ]
)

v = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

d_k = q.shape[-1]

# Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V
raw_scores = q @ k.T
scaled_scores = raw_scores / sqrt(d_k)
attention = torch.softmax(scaled_scores, dim=-1)
output = attention @ v

print("raw score shape:", tuple(raw_scores.shape))
print("attention shape:", tuple(attention.shape))
print("output shape:", tuple(output.shape))
print("\nattention matrix:")
print(attention)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

for ax, matrix, title in zip(
    axes,
    [raw_scores, scaled_scores, attention],
    ["QK^T", "QK^T / sqrt(d_k)", "softmax attention"],
):
    image = ax.imshow(matrix.numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=30)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    fig.colorbar(image, ax=ax, fraction=0.046)

fig.suptitle("각 Query token이 다른 token을 얼마나 참고하는가")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_scaled_dot_product_attention.png", dpi=160)
plt.show()
