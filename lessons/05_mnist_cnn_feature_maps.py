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

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "05_mnist_cnn_feature_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


# --------------------------------------------------
# 8. 고정된 test image 하나 선택
# --------------------------------------------------

sample_image, sample_label = test_dataset[0]

print("sample image shape:", sample_image.shape)
print("sample label:", sample_label)


# --------------------------------------------------
# 9. Feature map 추출 함수
# --------------------------------------------------

def get_feature_maps(model, image):

    model.eval()

    with torch.no_grad():
        x = image.unsqueeze(0)

        conv1 = model.conv1(x)
        relu1 = model.relu(conv1)
        pool1 = model.pool(relu1)

        conv2 = model.conv2(pool1)
        relu2 = model.relu(conv2)
        pool2 = model.pool(relu2)

    return relu1, pool1, relu2, pool2


# --------------------------------------------------
# 10. Feature map 저장 함수
# --------------------------------------------------

def save_feature_maps(
    feature_maps,
    title,
    filename,
    max_channels=8,
):

    feature_maps = feature_maps.squeeze(0)

    num_channels = min(
        feature_maps.size(0),
        max_channels,
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(10, 5),
    )

    axes = axes.flatten()

    for i in range(len(axes)):
        if i < num_channels:
            axes[i].imshow(
                feature_maps[i].cpu(),
                cmap="gray",
            )
            axes[i].set_title(f"channel {i}")

        axes[i].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved:", output_path)


# --------------------------------------------------
# 11. Conv1 filter 저장 함수
# --------------------------------------------------

def save_conv1_filters(model, title, filename):

    filters = model.conv1.weight.detach().cpu()

    fig, axes = plt.subplots(
        4,
        4,
        figsize=(7, 7),
    )

    axes = axes.flatten()

    for i in range(16):
        kernel = filters[i, 0]

        axes[i].imshow(
            kernel,
            cmap="gray",
        )
        axes[i].set_title(f"filter {i}")
        axes[i].axis("off")

    fig.suptitle(title)
    fig.tight_layout()

    output_path = OUTPUT_DIR / filename
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved:", output_path)


# --------------------------------------------------
# 12. 원본 이미지 저장 함수
# --------------------------------------------------

def save_original_image(image, label):

    fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(
        image.squeeze(),
        cmap="gray",
    )
    ax.set_title(f"Original Image - Label: {label}")
    ax.axis("off")

    fig.tight_layout()

    output_path = OUTPUT_DIR / "05_mnist_cnn_feature_maps_original_image.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("saved:", output_path)


# --------------------------------------------------
# 13. 학습 전 feature map 저장
# --------------------------------------------------

before_relu1, _, before_relu2, _ = get_feature_maps(
    model,
    sample_image,
)

save_original_image(
    sample_image,
    sample_label,
)

save_feature_maps(
    before_relu1,
    "Conv1 + ReLU - Before Training",
    "05_mnist_cnn_feature_maps_conv1_before_training.png",
)

save_feature_maps(
    before_relu2,
    "Conv2 + ReLU - Before Training",
    "05_mnist_cnn_feature_maps_conv2_before_training.png",
)

save_conv1_filters(
    model,
    "Conv1 Filters - Before Training",
    "05_mnist_cnn_feature_maps_conv1_filters_before_training.png",
)


# --------------------------------------------------
# 14. Training
# --------------------------------------------------

loss_history = []

for epoch in range(epochs):

    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        outputs = model(images)

        loss = criterion(
            outputs,
            labels,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    loss_history.append(average_loss)

    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
    )


# --------------------------------------------------
# 15. Test
# --------------------------------------------------

model.eval()

correct = 0
total = 0

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


accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")


# --------------------------------------------------
# 16. 학습 결과 텍스트 저장
# --------------------------------------------------

metrics_path = OUTPUT_DIR / "05_mnist_cnn_feature_maps_metrics.txt"

with metrics_path.open("w", encoding="utf-8") as f:
    f.write(f"epochs={epochs}\n")
    f.write(f"batch_size={batch_size}\n")
    f.write(f"learning_rate={learning_rate}\n")
    f.write(f"average_loss={loss_history[-1]:.6f}\n")
    f.write(f"test_accuracy={accuracy:.2f}%\n")

print("saved:", metrics_path)


# --------------------------------------------------
# 17. 학습 후 feature map 추출
# --------------------------------------------------

after_relu1, pool1, after_relu2, pool2 = get_feature_maps(
    model,
    sample_image,
)


# --------------------------------------------------
# 18. 학습 후 feature map / pooling / filter 저장
# --------------------------------------------------

save_feature_maps(
    after_relu1,
    "Conv1 + ReLU - After Training",
    "05_mnist_cnn_feature_maps_conv1_after_training.png",
)

save_feature_maps(
    after_relu2,
    "Conv2 + ReLU - After Training",
    "05_mnist_cnn_feature_maps_conv2_after_training.png",
)

save_feature_maps(
    pool1,
    "After First Max Pooling",
    "05_mnist_cnn_feature_maps_pool1_after_training.png",
)

save_feature_maps(
    pool2,
    "After Second Max Pooling",
    "05_mnist_cnn_feature_maps_pool2_after_training.png",
)

save_conv1_filters(
    model,
    "Conv1 Filters - After Training",
    "05_mnist_cnn_feature_maps_conv1_filters_after_training.png",
)
