"""ACT의 Action Chunking을 작은 trajectory로 직접 확인한다.

주의: 이 코드는 논문 성능 재현이 아니라 메커니즘 이해용 toy example이다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/08_act")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 하나의 demonstration action trajectory라고 생각할 수 있는 2차원 경로다.
time = torch.arange(20)
actions = torch.stack(
    [
        torch.sin(time.float() / 3.0),
        torch.cos(time.float() / 4.0),
    ],
    dim=1,
)

chunk_size = 5
query_times = [3, 7, 11]

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(actions[:, 0], actions[:, 1], "o-", label="full action trajectory")

for query_time in query_times:
    end = min(query_time + chunk_size, len(actions))
    chunk = actions[query_time:end]

    # ACT는 한 시점의 action 하나가 아니라 미래의 action 묶음을 한 번에 예측한다.
    ax.plot(
        chunk[:, 0],
        chunk[:, 1],
        "o-",
        linewidth=3,
        label=f"chunk from t={query_time}",
    )

ax.set_title("ACT: 한 번의 query에서 action chunk를 예측")
ax.set_xlabel("action dim 0")
ax.set_ylabel("action dim 1")
ax.legend()
ax.grid(True)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_action_chunking.png", dpi=160)
plt.show()
