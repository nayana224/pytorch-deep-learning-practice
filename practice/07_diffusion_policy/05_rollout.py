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
    raise SystemExit("Run 04_train.py first")

train_dataset = make_dataset()
policy, device = make_policy()

checkpoint = torch.load(checkpoint_path, map_location=device)
policy.load_state_dict(checkpoint["policy"])
policy.normalizer.load_state_dict(checkpoint["normalizer"])
policy.eval()

sample = train_dataset[0]

# Only the latest observation horizon is given to the policy.
observations = {
    key: value[:OBSERVATION_HORIZON].unsqueeze(0).to(device)
    for key, value in sample["obs"].items()
}

with torch.no_grad():
    result = policy.predict_action(observations)

full_prediction = result["action_pred"][0].cpu()
execution_chunk = result["action"][0].cpu()
demonstration = sample["action"]

print("full predicted horizon:", full_prediction.shape)
print("execution chunk:", execution_chunk.shape)
print("execute", ACTION_HORIZON, "actions, then re-plan")

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

ax.set_title("action prediction + receding horizon")
ax.set_aspect("equal", adjustable="box")
ax.legend()

plt.tight_layout()
plt.savefig(OUT / "05_receding_horizon.png", dpi=150)
plt.show()
