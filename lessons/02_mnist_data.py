from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# Output directory
# --------------------------------------------------

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "02_mnist_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. MNIST raw dataset (transform 없음)
# --------------------------------------------------

raw_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=None,
)

raw_tensor = raw_dataset.data[0]
raw_image, raw_label = raw_dataset[0]

print("raw tensor shape:", raw_tensor.shape)
print("raw tensor dtype:", raw_tensor.dtype)
print("raw PIL image type:", type(raw_image))
print("raw label:", raw_label)

# MNIST 클래스 내부의 원본 uint8 픽셀 데이터를 그대로 저장
raw_data_path = OUTPUT_DIR / "02_mnist_data_raw_uint8_tensor.pt"
torch.save(raw_tensor, raw_data_path)

# transform 이전 PIL 이미지를 PNG로 저장
raw_image_path = OUTPUT_DIR / "02_mnist_data_raw_pil_image.png"
raw_image.save(raw_image_path)


# --------------------------------------------------
# 2. MNIST -> Tensor
# --------------------------------------------------

transform = transforms.ToTensor()

tensor_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    tensor_dataset,
    batch_size=64,
    shuffle=True,
)

images, labels = next(iter(train_loader))

print("images:", images.shape)
print("labels:", labels.shape)
print("image dtype:", images.dtype)
print("label dtype:", labels.dtype)
print("labels:", labels[:10])
print("")


# --------------------------------------------------
# 3. ToTensor 적용 결과 1개 저장
# --------------------------------------------------

# raw sample과 같은 index(0)를 사용해서 변환 전/후를 비교하기 쉽게 한다.
tensor_image, tensor_label = tensor_dataset[0]

print("tensor image shape:", tensor_image.shape)
print("tensor image dtype:", tensor_image.dtype)
print("tensor label:", tensor_label)

# Tensor 자체도 보존한다.
tensor_data_path = OUTPUT_DIR / "02_mnist_data_transformed_tensor.pt"
torch.save(tensor_image, tensor_data_path)

# 사람이 바로 확인할 수 있도록 Tensor를 이미지로 시각화해서 저장한다.
tensor_image_path = OUTPUT_DIR / "02_mnist_data_transformed_tensor_image.png"

plt.figure(figsize=(4, 4))
plt.imshow(tensor_image.squeeze(), cmap="gray")
plt.title(f"Tensor image - label={tensor_label}")
plt.axis("off")
plt.tight_layout()
plt.savefig(tensor_image_path, dpi=150, bbox_inches="tight")
plt.close()


print("saved outputs:")
print(" -", raw_data_path)
print(" -", raw_image_path)
print(" -", tensor_data_path)
print(" -", tensor_image_path)
