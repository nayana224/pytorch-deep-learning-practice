"""Q/K/V projection을 작은 예제로 직접 확인한다.

전체 Transformer를 구현하지 않고, self-attention의 입력이
어떻게 Query, Key, Value로 바뀌는지 shape과 값으로 확인한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/04_attention_is_all_you_need")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 논문의 핵심 연산만 보기 위해 4개의 token과 작은 embedding을 사용한다.
tokens = ["I", "study", "deep", "learning"]

# 각 행이 하나의 token embedding이다. [token, d_model]
x = torch.tensor(
    [
        [1.0, 0.0, 1.0, 0.0],
        [0.0, 2.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 2.0, 1.0],
    ],
    dtype=torch.float32,
)

# 학습되는 projection matrix를 작은 고정 행렬로 대신한다.
# 목적은 "Q/K/V가 서로 다른 관점의 projection"이라는 점을 보는 것이다.
w_q = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.5],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 0.5, 0.0, 1.0],
    ]
)

w_k = torch.tensor(
    [
        [0.5, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.5],
        [1.0, 0.0, 0.5, 0.0],
        [0.0, 0.5, 0.0, 1.0],
    ]
)

w_v = torch.tensor(
    [
        [1.0, 0.5, 0.0, 0.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.0, 0.0, 1.0, 0.5],
        [0.5, 0.0, 0.0, 1.0],
    ]
)

q = x @ w_q
k = x @ w_k
v = x @ w_v

print("X shape:", tuple(x.shape))
print("Q shape:", tuple(q.shape))
print("K shape:", tuple(k.shape))
print("V shape:", tuple(v.shape))

fig, axes = plt.subplots(1, 4, figsize=(14, 4))

for ax, matrix, title in zip(
    axes,
    [x, q, k, v],
    ["Input X", "Query Q", "Key K", "Value V"],
):
    image = ax.imshow(matrix.numpy(), aspect="auto")
    ax.set_title(title)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens)
    ax.set_xlabel("feature dimension")
    fig.colorbar(image, ax=ax, fraction=0.046)

fig.suptitle("The same token embeddings are projected into Q, K, and V")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_qkv_projection.png", dpi=160)
plt.show()
