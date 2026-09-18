"""Atrous convolution의 sampling 간격을 가장 작은 예제로 확인한다.

DeepLab 계열의 핵심은 kernel parameter 수를 늘리지 않고
dilation을 사용해 더 넓은 spatial context를 보는 것이다.
이 파일은 segmentation 학습이 아니라 그 sampling pattern 자체를 확인한다.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


OUTPUT_DIR = Path("outputs/03_deeplabv3plus")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 중앙 pixel 하나만 1로 두면 3x3 kernel이 어떤 위치를 샘플링하는지
# 출력 pattern에서 바로 확인할 수 있다.
x = torch.zeros(1, 1, 31, 31)
x[0, 0, 15, 15] = 1.0

# 모든 weight를 1로 둬서 학습된 filter 효과는 제거한다.
# 여기서는 오직 dilation에 따른 sampling 위치 차이만 본다.
kernel = torch.ones(1, 1, 3, 3)
dilation_rates = [1, 2, 4, 6]

fig, axes = plt.subplots(1, 4, figsize=(12, 3))

for ax, rate in zip(axes, dilation_rates):
    y = F.conv2d(
        x,
        kernel,
        padding=rate,
        dilation=rate,
    )

    ax.imshow(y[0, 0], cmap="viridis")
    ax.set_title(f"dilation={rate}")
    ax.axis("off")

fig.suptitle("Atrous convolution: wider sampling with the same 3x3 kernel")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "02_atrous_receptive_field.png", dpi=150)
plt.show()
