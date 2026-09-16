import matplotlib.pyplot as plt
import torch

from common import OBSERVATION_HORIZON, OUT, make_dataset, make_policy


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
observations = {
    key: value[:OBSERVATION_HORIZON].unsqueeze(0).to(device)
    for key, value in sample["obs"].items()
}

# Diffusion sampling is stochastic. Re-sample several action sequences
# from the same observation to inspect multimodality / spread.
predictions = []

with torch.no_grad():
    for _ in range(8):
        result = policy.predict_action(observations)
        predictions.append(result["action_pred"][0].cpu())

fig, ax = plt.subplots(figsize=(7, 6))

for index, prediction in enumerate(predictions):
    label = f"sample {index}" if index < 3 else None
    ax.plot(
        prediction[:, 0],
        prediction[:, 1],
        "o-",
        alpha=0.55,
        label=label,
    )

demonstration = sample["action"]
ax.plot(
    demonstration[:, 0],
    demonstration[:, 1],
    "k--",
    linewidth=2,
    label="demonstration",
)

ax.set_title("stochastic action samples")
ax.set_aspect("equal", adjustable="box")
ax.legend()

plt.tight_layout()
plt.savefig(OUT / "06_multimodal_samples.png", dpi=150)
plt.show()

stacked = torch.stack(predictions)
mean_sample_std = stacked.std(dim=0).mean().item()

step_distances = []
for prediction in predictions:
    delta = prediction[1:] - prediction[:-1]
    step_distances.append(delta.norm(dim=-1).mean())

mean_step_distance = torch.stack(step_distances).mean().item()

print("mean sample std:", mean_sample_std)
print("mean action-step distance:", mean_step_distance)
