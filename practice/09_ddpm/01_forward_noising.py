"""DDPM forward process에서 이미지에 점점 noise가 추가되는 과정을 확인한다."""

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

to_tensor = transforms.ToTensor()
image, label = dataset[0]
x0 = to_tensor(image)

# 논문과 같은 선형 beta schedule의 핵심 형태를 사용한다.
num_steps = 1000
beta = torch.linspace(1e-4, 0.02, num_steps)
alpha = 1.0 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

# 같은 epsilon을 사용하면 timestep 변화만 비교하기 쉽다.
torch.manual_seed(0)
epsilon = torch.randn_like(x0)

timesteps = [0, 99, 299, 499, 799, 999]
noisy_images = []

for timestep in timesteps:
    a_bar = alpha_bar[timestep]

    # q(x_t | x_0) = sqrt(alpha_bar_t) x_0
    #                  + sqrt(1-alpha_bar_t) epsilon
    xt = (
        torch.sqrt(a_bar) * x0
        + torch.sqrt(1.0 - a_bar) * epsilon
    )
    noisy_images.append(xt.clamp(0, 1))

fig, axes = plt.subplots(1, len(timesteps), figsize=(18, 3.5))

for ax, noisy, timestep in zip(axes, noisy_images, timesteps):
    ax.imshow(noisy.permute(1, 2, 0))
    ax.set_title(f"t={timestep}")
    ax.axis("off")

fig.suptitle(
    "DDPM forward process: timestep이 커질수록 x0 정보가 noise로 사라진다"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "01_forward_noising.png", dpi=160)
plt.show()

print("class:", dataset.classes[label])
print("x0 shape:", tuple(x0.shape))
print("epsilon shape:", tuple(epsilon.shape))
