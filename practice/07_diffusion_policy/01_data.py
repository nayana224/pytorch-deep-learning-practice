import matplotlib.pyplot as plt

from common import (
    ACTION_HORIZON,
    OBSERVATION_HORIZON,
    OUT,
    PREDICTION_HORIZON,
    make_dataset,
)


# Official Push-T demonstration dataset
train_dataset = make_dataset()
sample = train_dataset[0]

image_sequence = sample["obs"]["image"]
agent_positions = sample["obs"]["agent_pos"]
action_sequence = sample["action"]

print("dataset samples      :", len(train_dataset))
print("image sequence       :", image_sequence.shape)
print("agent positions      :", agent_positions.shape)
print("action sequence      :", action_sequence.shape)
print("prediction horizon   :", PREDICTION_HORIZON)
print("observation horizon  :", OBSERVATION_HORIZON)
print("action horizon       :", ACTION_HORIZON)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].imshow(image_sequence[0].permute(1, 2, 0).clamp(0, 1))
axes[0].set_title("observation 0")

axes[1].imshow(image_sequence[1].permute(1, 2, 0).clamp(0, 1))
axes[1].set_title("observation 1")

axes[2].plot(
    action_sequence[:, 0],
    action_sequence[:, 1],
    "o-",
)
axes[2].scatter(
    agent_positions[:OBSERVATION_HORIZON, 0],
    agent_positions[:OBSERVATION_HORIZON, 1],
    marker="x",
    s=80,
)
axes[2].set_title("predicted-action horizon target")
axes[2].set_aspect("equal", adjustable="box")

axes[0].axis("off")
axes[1].axis("off")

plt.tight_layout()
plt.savefig(OUT / "01_pusht_data.png", dpi=150)
plt.show()
