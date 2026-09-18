"""Diffusion Policy의 receding-horizon 실행 방식을 확인한다.

policy는 긴 future action horizon을 한 번에 예측하지만,
실제로는 앞부분의 action만 실행한 뒤 새 observation으로 다시 계획한다.
"""

import matplotlib.pyplot as plt
import torch

from common import (
    ACTION_HORIZON,
    OBSERVATION_HORIZON,
    OUT,
    make_dataset,
    make_policy,
)


checkpoint_path = OUT / "scaled_policy.pt"
if not checkpoint_path.exists():
    raise SystemExit(
        "scaled checkpoint가 없습니다. 먼저 실행하세요:\n"
        "  python practice/10_diffusion_policy/04_train.py --epochs 10"
    )

train_dataset = make_dataset()
policy, device = make_policy()

checkpoint = torch.load(checkpoint_path, map_location=device)
policy.load_state_dict(checkpoint["policy"])
policy.normalizer.load_state_dict(checkpoint["normalizer"])
policy.eval()

sample = train_dataset[0]

# observation horizon만큼의 최근 관측만 policy condition으로 넣는다.
observations = {
    key: value[:OBSERVATION_HORIZON].unsqueeze(0).to(device)
    for key, value in sample["obs"].items()
}

with torch.no_grad():
    result = policy.predict_action(observations)

# action_pred: model이 예측한 전체 future horizon
# action: 그중 지금 실제로 실행할 앞부분의 action chunk
full_prediction = result["action_pred"][0].cpu()
execution_chunk = result["action"][0].cpu()
demonstration = sample["action"]

print("full predicted horizon:", tuple(full_prediction.shape))
print("execution chunk:", tuple(execution_chunk.shape))
print(
    f"{ACTION_HORIZON}개 action을 실행한 뒤 새 observation으로 다시 계획"
)

fig, ax = plt.subplots(figsize=(6, 5))

ax.plot(
    demonstration[:, 0],
    demonstration[:, 1],
    "o-",
    label="demonstration",
)
ax.plot(
    full_prediction[:, 0],
    full_prediction[:, 1],
    "o-",
    label="predicted horizon",
)
ax.plot(
    execution_chunk[:, 0],
    execution_chunk[:, 1],
    "o-",
    linewidth=3,
    label="execute now",
)

ax.set_title("Diffusion Policy: prediction horizon vs execution horizon")
ax.set_aspect("equal", adjustable="box")
ax.legend()

plt.tight_layout()
plt.savefig(OUT / "05_receding_horizon.png", dpi=150)
plt.show()
