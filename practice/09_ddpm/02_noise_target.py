"""DDPM 학습에서 모델이 무엇을 예측하는지 확인한다.

모델의 target은 clean image 자체가 아니라
x_t를 만들 때 사용한 Gaussian noise epsilon이다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10


DATA_DIR = Path("data/09_ddpm")
OUTPUT_DIR = Path("outputs/09_ddpm")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

try:
    dataset = CIFAR10(DATA_DIR, train=True, download=False)
except RuntimeError as error:
    raise FileNotFoundError(
        "CIFAR-10이 없습니다. 먼저 실행하세요:\n"
        "  python scripts/download_torchvision_data.py ddpm"
    ) from error

x0 = transforms.ToTensor()(dataset[0][0])

num_steps = 1000
beta = torch.linspace(1e-4, 0.02, num_steps)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

timestep = 499
torch.manual_seed(1)
epsilon = torch.randn_like(x0)

a_bar = alpha_bar[timestep]
xt = (
    torch.sqrt(a_bar) * x0
    + torch.sqrt(1.0 - a_bar) * epsilon
)

# 실제 학습에서는 epsilon_theta(x_t, t)가 이 epsilon을 맞추도록 MSE를 최소화한다.
# 여기서는 "완벽한 예측"과 "틀린 예측"의 loss 차이만 확인한다.
perfect_prediction = epsilon.clone()
wrong_prediction = torch.zeros_like(epsilon)

perfect_loss = torch.mean((perfect_prediction - epsilon) ** 2)
wrong_loss = torch.mean((wrong_prediction - epsilon) ** 2)

print("timestep:", timestep)
print("perfect epsilon MSE:", perfect_loss.item())
print("zero prediction MSE:", wrong_loss.item())

# noise는 음수/양수를 가지므로 보기 쉽게 채널 평균 후 범위를 정규화한다.
noise_vis = epsilon.mean(dim=0)
noise_vis = noise_vis - noise_vis.min()
noise_vis = noise_vis / noise_vis.max().clamp_min(1e-8)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

axes[0].imshow(x0.permute(1, 2, 0))
axes[0].set_title("clean x0")

axes[1].imshow(xt.clamp(0, 1).permute(1, 2, 0))
axes[1].set_title(f"noisy x_t, t={timestep}")

axes[2].imshow(noise_vis, cmap="gray")
axes[2].set_title("training target epsilon")

for ax in axes:
    ax.axis("off")

fig.suptitle("DDPM training target: predict the added noise epsilon")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_noise_prediction_target.png", dpi=160)
plt.show()
