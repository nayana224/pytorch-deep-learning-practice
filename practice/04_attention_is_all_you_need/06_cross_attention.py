"""Encoder-Decoder Cross-Attention의 Q/K/V 출처를 직접 확인한다.

Self-Attention과 달리 Cross-Attention에서는
Q는 decoder hidden state에서,
K와 V는 encoder output에서 만들어진다.
"""

from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/04_attention_is_all_you_need")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

source_tokens = ["I", "study", "deep", "learning"]
target_tokens = ["Je", "etudie", "IA"]

# encoder가 source 문장을 처리한 결과라고 가정한다.
encoder_output = torch.tensor(
    [
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 1.5, 0.0, 0.5],
        [0.5, 0.5, 1.0, 0.0],
        [0.0, 1.0, 1.5, 1.0],
    ],
    dtype=torch.float32,
)

# decoder의 현재 hidden state라고 가정한다.
decoder_hidden = torch.tensor(
    [
        [1.0, 0.2, 0.1, 0.0],
        [0.2, 0.8, 1.0, 0.1],
        [0.0, 1.2, 0.4, 1.0],
    ],
    dtype=torch.float32,
)

# 실제 모델에서는 아래 행렬들이 학습된다.
# 여기서는 Q/K/V가 어디에서 오는지 확인하는 것이 목적이라 identity projection을 쓴다.
q = decoder_hidden
k = encoder_output
v = encoder_output

scores = (q @ k.T) / sqrt(q.shape[-1])
attention = torch.softmax(scores, dim=-1)
context = attention @ v

print("Q from decoder shape:", tuple(q.shape))
print("K from encoder shape:", tuple(k.shape))
print("V from encoder shape:", tuple(v.shape))
print("cross-attention shape:", tuple(attention.shape))
print("context shape:", tuple(context.shape))

fig, ax = plt.subplots(figsize=(7, 5))
image = ax.imshow(
    attention.numpy(),
    vmin=0,
    vmax=1,
    aspect="auto",
)

ax.set_xticks(range(len(source_tokens)))
ax.set_xticklabels(source_tokens, rotation=30)
ax.set_yticks(range(len(target_tokens)))
ax.set_yticklabels(target_tokens)
ax.set_xlabel("Encoder Key / Value token")
ax.set_ylabel("Decoder Query token")
ax.set_title("Encoder-Decoder Cross-Attention")

fig.colorbar(image, ax=ax, fraction=0.046)
fig.suptitle("Which encoder source tokens each decoder query attends to")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_cross_attention.png", dpi=160)
plt.show()
