"""Multi-Head Attention에서 head마다 다른 관계를 볼 수 있음을 확인한다."""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/04_attention_is_all_you_need")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

torch.manual_seed(7)

tokens = ["I", "study", "deep", "learning"]
num_tokens = len(tokens)
d_model = 8
num_heads = 4
d_head = d_model // num_heads

# 하나의 입력 sequence를 여러 head가 서로 다른 projection으로 본다.
x = torch.randn(num_tokens, d_model)

w_q = torch.randn(num_heads, d_model, d_head)
w_k = torch.randn(num_heads, d_model, d_head)

attention_maps = []

for head in range(num_heads):
    q = x @ w_q[head]
    k = x @ w_k[head]

    scores = (q @ k.T) / sqrt(d_head)
    attention = torch.softmax(scores, dim=-1)
    attention_maps.append(attention)

fig, axes = plt.subplots(1, num_heads, figsize=(16, 4))

for head, (ax, attention) in enumerate(zip(axes, attention_maps)):
    image = ax.imshow(
        attention.numpy(),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_title(f"Head {head + 1}")
    ax.set_xticks(range(num_tokens))
    ax.set_xticklabels(tokens, rotation=30)
    ax.set_yticks(range(num_tokens))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    fig.colorbar(image, ax=ax, fraction=0.046)

fig.suptitle("서로 다른 head는 동일한 입력에서도 다른 attention pattern을 만든다")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "04_multihead_attention.png", dpi=160)
plt.show()
