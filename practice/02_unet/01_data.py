from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile
import torch
import torch.nn as nn


DATA_DIR = Path("data/02_unet/isbi2012")

image_path = DATA_DIR / "train-volume.tif"
label_path = DATA_DIR / "train-labels.tif"


# 1. TIFF stack 읽기
images = tifffile.imread(image_path)
labels = tifffile.imread(label_path)

print("images shape:", images.shape)
print("images dtype:", images.dtype)
print("images min:", images.min())
print("images max:", images.max())

print()

print("labels shape:", labels.shape)
print("labels dtype:", labels.dtype)
print("labels min:", labels.min())
print("labels max:", labels.max())
print("labels unique:", np.unique(labels))


# 2. 한 장 선택
index = 0

image = images[index]
label = labels[index]


# 3. 원본 image / GT 확인
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image, cmap="gray")
axes[0].set_title("EM image")
axes[0].axis("off")

axes[1].imshow(label, cmap="gray")
axes[1].set_title("GT mask")
axes[1].axis("off")

axes[2].imshow(image, cmap="gray")
axes[2].imshow(label == 0, alpha=0.4, cmap="Reds")
axes[2].set_title("Overlay: membrane")
axes[2].axis("off")

plt.tight_layout()
plt.show()


# 4. NumPy -> PyTorch tensor
image_tensor = torch.from_numpy(image).float() / 255.0
image_tensor = image_tensor.unsqueeze(0)

label_tensor = torch.from_numpy(label)
label_tensor = (label_tensor == 255).long()


print()
print("image tensor shape:", image_tensor.shape)
print("image tensor dtype:", image_tensor.dtype)

print("label tensor shape:", label_tensor.shape)
print("label tensor dtype:", label_tensor.dtype)
print("label tensor unique:", torch.unique(label_tensor))


# 5. batch 차원 추가
input_batch = image_tensor.unsqueeze(0)
target_batch = label_tensor.unsqueeze(0)

print()
print("input batch shape:", input_batch.shape)
print("target batch shape:", target_batch.shape)


# 6. U-Net 출력이라고 가정한 dummy logits
dummy_logits = torch.randn(1, 2, 512, 512)

print("dummy logits shape:", dummy_logits.shape)


# 7. CrossEntropyLoss 연결 확인
criterion = nn.CrossEntropyLoss()

loss = criterion(dummy_logits, target_batch)

print("loss:", loss.item())