"""정답 epsilon을 안다고 가정했을 때 x_t에서 x_0를 복원해본다.

이 실습은 reverse diffusion network를 학습하는 것이 아니라,
왜 epsilon prediction이 clean sample 복원과 연결되는지 수식으로 확인한다.
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

timestep = 699
torch.manual_seed(2)
epsilon = torch.randn_like(x0)

a_bar = alpha_bar[timestep]
xt = (
    torch.sqrt(a_bar) * x0
    + torch.sqrt(1.0 - a_bar) * epsilon
)

# forward 식을 x0에 대해 정리하면,
# 정답 epsilon을 알고 있을 때 clean x0를 계산할 수 있다.
x0_reconstructed = (
    xt - torch.sqrt(1.0 - a_bar) * epsilon
) / torch.sqrt(a_bar)

reconstruction_error = torch.mean(
    (x0_reconstructed - x0) ** 2
).item()

print("reconstruction MSE with true epsilon:", reconstruction_error)

fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

axes[0].imshow(x0.permute(1, 2, 0))
axes[0].set_title("original x0")

axes[1].imshow(xt.clamp(0, 1).permute(1, 2, 0))
axes[1].set_title(f"x_t, t={timestep}")

axes[2].imshow(x0_reconstructed.clamp(0, 1).permute(1, 2, 0))
axes[2].set_title("x0 from true epsilon")

for ax in axes:
    ax.axis("off")

fig.suptitle("noise를 맞게 예측하면 noisy sample에서 clean signal을 추정할 수 있다")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "03_reconstruct_x0.png", dpi=160)
plt.show()
