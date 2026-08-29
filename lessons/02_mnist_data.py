import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# PyTorch의 tensor 형태로 객체 생성
transform = transforms.ToTensor()

# 그러면 원본은 어떻게 생겼길래?
train_dataset = datasets.MNIST(
    root = "./data",
    train=True,
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_dataset,
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


# 이미지를 직접 본다.
import matplotlib.pyplot as plt

image = images[0]
label = labels[0]

print(images.shape)
print(label)

plt.imshow(image.squeeze(), cmap="gray")
plt.title(f"label={label.item()}")
plt.show()
print("")



