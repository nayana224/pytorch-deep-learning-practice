"""Diffusion Policy에서 robot action sequence 자체에 noise를 넣는 과정을 본다.

DDPM의 image x가 Diffusion Policy에서는 future action chunk가 된다는 점이 핵심이다.
이 파일은 실제 Push-T demonstration을 사용해 clean action → noisy action 변화를 확인한다.
"""

import matplotlib.pyplot as plt
import torch

from common import OUT, make_dataset, make_scheduler


# 1) 실제 Push-T demonstration의 clean action sequence를 가져온다.
train_dataset = make_dataset()
normalizer = train_dataset.get_normalizer()

action = train_dataset[0]["action"].unsqueeze(0)
normalized_action = normalizer["action"].normalize(action)

# 2) 논문/공개 구현에서 사용하는 DDPM scheduler를 준비한다.
scheduler = make_scheduler()

# 학습 시 모델이 맞혀야 하는 target은 아래 Gaussian noise epsilon이다.
noise = torch.randn_like(normalized_action)

timesteps = [0, 20, 50, 99]

fig, axes = plt.subplots(
    1,
    len(timesteps) + 1,
    figsize=(16, 3),
)

axes[0].plot(
    normalized_action[0, :, 0],
    normalized_action[0, :, 1],
    "o-",
)
axes[0].set_title("clean action chunk")

for ax, timestep in zip(axes[1:], timesteps):
    t = torch.tensor([timestep], dtype=torch.long)

    # DDPM forward process와 동일하게 action sequence에 noise를 추가한다.
    noisy_action = scheduler.add_noise(
        normalized_action,
        noise,
        t,
    )

    ax.plot(
        noisy_action[0, :, 0],
        noisy_action[0, :, 1],
        "o-",
    )
    ax.set_title(f"t={timestep}")

for ax in axes:
    ax.set_aspect("equal", adjustable="box")

fig.suptitle("Diffusion Policy: clean robot action → noisy action sequence")
fig.tight_layout()
fig.savefig(OUT / "02_action_diffusion.png", dpi=150)
plt.show()

print("clean action shape:", tuple(normalized_action.shape))
print("epsilon target shape:", tuple(noise.shape))
print("학습 target: sampled Gaussian noise epsilon")
