"""Decoder의 causal mask가 미래 token attention을 막는 과정을 확인한다."""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/04_attention_is_all_you_need")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

tokens = ["I", "study", "deep", "learning"]

x = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0, 1.0],
    ]
)

scores = (x @ x.T) / sqrt(x.shape[-1])
unmasked_attention = torch.softmax(scores, dim=-1)

# upper triangular 영역이 현재 token보다 "미래"에 해당한다.
future_mask = torch.triu(
    torch.ones_like(scores, dtype=torch.bool),
    diagonal=1,
)

masked_scores = scores.masked_fill(future_mask, float("-inf"))
masked_attention = torch.softmax(masked_scores, dim=-1)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, matrix, title in zip(
    axes,
    [unmasked_attention, masked_attention],
    ["Self-Attention", "Masked Self-Attention"],
):
    image = ax.imshow(matrix.numpy(), vmin=0, vmax=1, aspect="auto")
    ax.set_title(title)
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=30)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    fig.colorbar(image, ax=ax, fraction=0.046)

fig.suptitle("Causal masking prevents attention to future tokens")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_masked_vs_unmasked.png", dpi=160)
plt.show()
