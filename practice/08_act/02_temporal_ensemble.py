"""ACT의 temporal ensembling을 겹치는 action chunk로 확인한다.

여러 시점에서 예측한 chunk가 현재 action에 대해 서로 다른 값을 줄 때,
최근 prediction에 더 큰 가중치를 주어 합치는 과정을 시각화한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/08_act")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

current_time = 8

# 서로 다른 과거 query에서 생성된 "현재 시점 action" 후보라고 생각한다.
predictions = torch.tensor(
    [
        [0.20, 0.80],
        [0.35, 0.70],
        [0.55, 0.60],
        [0.70, 0.45],
    ],
    dtype=torch.float32,
)

# 오래된 prediction에는 작은 weight, 최근 prediction에는 큰 weight를 준다.
ages = torch.tensor([3.0, 2.0, 1.0, 0.0])
decay = 0.8
weights = torch.exp(-decay * ages)
weights = weights / weights.sum()

ensemble = (predictions * weights[:, None]).sum(dim=0)

print("weights:", weights)
print("temporal ensemble action:", ensemble)

fig, ax = plt.subplots(figsize=(7, 6))

for index, prediction in enumerate(predictions):
    ax.scatter(
        prediction[0],
        prediction[1],
        s=120,
        label=f"prediction {index}, w={weights[index]:.2f}",
    )

ax.scatter(
    ensemble[0],
    ensemble[1],
    s=220,
    marker="X",
    label="temporal ensemble",
)

ax.set_title(f"ACT temporal ensembling at t={current_time}")
ax.set_xlabel("action dim 0")
ax.set_ylabel("action dim 1")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_temporal_ensemble.png", dpi=160)
plt.show()
