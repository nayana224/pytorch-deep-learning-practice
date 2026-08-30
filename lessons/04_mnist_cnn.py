from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# --------------------------------------------------
# 1. Hyperparameters
# --------------------------------------------------

batch_size = 64
learning_rate = 0.001
epochs = 1


# --------------------------------------------------
# 2. Output directory
# --------------------------------------------------

output_dir = Path("outputs/04_mnist_cnn")
output_dir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 3. MNIST -> Tensor
# --------------------------------------------------

transform = transforms.ToTensor()


# --------------------------------------------------
# 4. Dataset
# --------------------------------------------------

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


# --------------------------------------------------
# 5. DataLoader
# --------------------------------------------------

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


# --------------------------------------------------
# 6. CNN Model
# --------------------------------------------------

class CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1,
        )

        self.fc = nn.Linear(
            32 * 7 * 7,
            10,
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


model = CNN()


# --------------------------------------------------
# 7. Loss / Optimizer
# --------------------------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)


# --------------------------------------------------
# 8. Training
# --------------------------------------------------

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


# --------------------------------------------------
# 9. Test + prediction collection
# --------------------------------------------------

model.eval()

correct = 0
total = 0

all_labels = []
all_predictions = []
misclassified_samples = []

with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(images)

        _, predicted = torch.max(
            outputs,
            dim=1,
        )

        total += labels.size(0)
        correct += (
            predicted == labels
        ).sum().item()

        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predicted.cpu().tolist())

        if len(misclassified_samples) < 8:
            wrong_indices = (
                predicted != labels
            ).nonzero(as_tuple=True)[0]

            for index in wrong_indices:
                if len(misclassified_samples) >= 8:
                    break

                misclassified_samples.append(
                    (
                        images[index].cpu(),
                        labels[index].item(),
                        predicted[index].item(),
                    )
                )


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Parameter Count: {parameter_count:,}")


# --------------------------------------------------
# 10. Confusion matrix
# --------------------------------------------------

num_classes = 10
confusion_matrix = torch.zeros(
    num_classes,
    num_classes,
    dtype=torch.int64,
)

for true_label, predicted_label in zip(
    all_labels,
    all_predictions,
):
    confusion_matrix[
        true_label,
        predicted_label,
    ] += 1

fig, ax = plt.subplots(figsize=(8, 7))

image = ax.imshow(
    confusion_matrix.numpy(),
    cmap="Blues",
)

ax.set_title("CNN Confusion Matrix")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_xticks(range(num_classes))
ax.set_yticks(range(num_classes))

for row in range(num_classes):
    for col in range(num_classes):
        ax.text(
            col,
            row,
            str(confusion_matrix[row, col].item()),
            ha="center",
            va="center",
            fontsize=7,
        )

fig.colorbar(image, ax=ax)
fig.tight_layout()
fig.savefig(
    output_dir / "04_mnist_cnn_confusion_matrix.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# 11. Misclassified samples
# --------------------------------------------------

fig, axes = plt.subplots(
    2,
    4,
    figsize=(10, 5),
)

axes = axes.flatten()

for index, axis in enumerate(axes):

    if index < len(misclassified_samples):
        image_tensor, true_label, predicted_label = (
            misclassified_samples[index]
        )

        axis.imshow(
            image_tensor.squeeze(),
            cmap="gray",
        )

        axis.set_title(
            f"True: {true_label} / Pred: {predicted_label}"
        )

    axis.axis("off")

fig.suptitle("CNN Misclassified Samples")
fig.tight_layout()
fig.savefig(
    output_dir / "04_mnist_cnn_misclassified_samples.png",
    dpi=200,
    bbox_inches="tight",
)
plt.close(fig)


# --------------------------------------------------
# 12. Save metrics
# --------------------------------------------------

metrics_path = output_dir / "04_mnist_cnn_metrics.txt"

with metrics_path.open("w", encoding="utf-8") as file:
    file.write(f"epochs={epochs}\n")
    file.write(f"batch_size={batch_size}\n")
    file.write(f"learning_rate={learning_rate}\n")
    file.write("loss_function=CrossEntropyLoss\n")
    file.write("optimizer=Adam\n")
    file.write(f"parameter_count={parameter_count}\n")
    file.write(f"average_training_loss={average_loss:.6f}\n")
    file.write(f"test_accuracy={accuracy:.2f}%\n")

print(f"Saved outputs to: {output_dir}")
