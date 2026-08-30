import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# hyperparameter
batch_size = 64
learning_rate = 0.001
epochs = 1

# output directory
output_dir = "./outputs/03_mnist_mlp"
os.makedirs(output_dir, exist_ok=True)

# MNIST를 Tensor로 변환
transform = transforms.ToTensor()

# Dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
)

# MLP 모델
class MLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # [B, 1, 28, 28] -> [B, 784]
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = MLP()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# Training
for epoch in range(epochs):
    model.train()

    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )

# Test + 결과 수집
model.eval()

correct = 0
total = 0
confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
misclassified_samples = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        for actual, pred in zip(labels, predicted):
            confusion_matrix[actual.item(), pred.item()] += 1

        wrong_indices = (predicted != labels).nonzero(as_tuple=True)[0]

        for idx in wrong_indices:
            if len(misclassified_samples) >= 8:
                break

            misclassified_samples.append(
                (
                    images[idx].cpu(),
                    labels[idx].item(),
                    predicted[idx].item(),
                )
            )

accuracy = 100 * correct / total
parameter_count = sum(p.numel() for p in model.parameters())

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Parameter Count: {parameter_count}")

# Confusion Matrix 저장
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(confusion_matrix.numpy(), cmap="Blues")

ax.set_title("MLP Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(range(10))
ax.set_yticks(range(10))

for i in range(10):
    for j in range(10):
        value = confusion_matrix[i, j].item()
        ax.text(j, i, str(value), ha="center", va="center", fontsize=7)

fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "03_mnist_mlp_confusion_matrix.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# 오분류 샘플 저장
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
axes = axes.flatten()

for i, ax in enumerate(axes):
    if i < len(misclassified_samples):
        image, actual, pred = misclassified_samples[i]
        ax.imshow(image.squeeze(), cmap="gray")
        ax.set_title(f"True: {actual} / Pred: {pred}")

    ax.axis("off")

plt.suptitle("MLP Misclassified Samples")
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "03_mnist_mlp_misclassified_samples.png"),
    dpi=200,
    bbox_inches="tight",
)
plt.close()

# 학습/평가 결과 텍스트 저장
metrics_path = os.path.join(output_dir, "03_mnist_mlp_metrics.txt")

with open(metrics_path, "w", encoding="utf-8") as f:
    f.write(f"epochs={epochs}\n")
    f.write(f"batch_size={batch_size}\n")
    f.write(f"learning_rate={learning_rate}\n")
    f.write("loss_function=CrossEntropyLoss\n")
    f.write("optimizer=Adam\n")
    f.write(f"parameter_count={parameter_count}\n")
    f.write(f"average_training_loss={average_loss:.6f}\n")
    f.write(f"test_accuracy={accuracy:.2f}%\n")

print(f"Saved outputs to: {output_dir}")
