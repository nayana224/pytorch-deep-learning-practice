from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


OUTPUT_DIR = Path("outputs/05_mnist_cnn")


class MnistCnn(nn.Module):
    """Convolution과 pooling으로 MNIST를 분류하는 작은 CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """첫 번째와 두 번째 convolution block의 feature map을 반환한다."""
        feature1 = self.relu(self.conv1(x))
        x = self.pool(feature1)
        feature2 = self.relu(self.conv2(x))
        x = self.pool(feature2)
        return feature1, feature2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, feature2 = self.forward_features(x)
        x = self.pool(feature2)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    """전체 test set에서 accuracy를 계산한다."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()

    return correct / total


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        "./data",
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    model = MnistCnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fixed_image, fixed_label = test_dataset[0]
    fixed_image = fixed_image.unsqueeze(0).to(device)
    print(f"fixed sample label: {fixed_label}")

    loss_history: list[float] = []
    accuracy_history: list[float] = []

    visualize_feature_maps(model, fixed_image, epoch=0)
    visualize_conv1_filters(model, epoch=0)

    with torch.no_grad():
        feature1, feature2 = model.forward_features(fixed_image)
    print(f"input: {tuple(fixed_image.shape)}")
    print(f"conv1 feature map: {tuple(feature1.shape)}")
    print(f"conv2 feature map: {tuple(feature2.shape)}")

    for epoch in range(5):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        mean_loss = running_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)
        loss_history.append(mean_loss)
        accuracy_history.append(accuracy)

        print(
            f"epoch={epoch + 1} "
            f"loss={mean_loss:.4f} "
            f"test_accuracy={accuracy:.4f}"
        )

        visualize_feature_maps(model, fixed_image, epoch=epoch + 1)
        visualize_conv1_filters(model, epoch=epoch + 1)

    plot_training_curves(loss_history, accuracy_history)
    print(f"시각화 저장 위치: {OUTPUT_DIR}")
    print("같은 입력 이미지의 feature map을 epoch별로 비교하세요.")
    plt.show()


def visualize_feature_maps(
    model: MnistCnn,
    image: torch.Tensor,
    epoch: int,
) -> None:
    """동일한 입력에 대한 CNN feature map을 epoch별로 저장한다."""
    model.eval()
    with torch.no_grad():
        feature1, feature2 = model.forward_features(image)

    feature1 = feature1[0].cpu()
    feature2 = feature2[0].cpu()

    figure1, axes1 = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes1.flat):
        axis.imshow(feature1[index].numpy(), cmap="gray")
        axis.set_title(f"ch {index}")
        axis.axis("off")
    figure1.suptitle(f"Conv1 feature maps - epoch {epoch}")
    figure1.tight_layout()
    figure1.savefig(
        OUTPUT_DIR / f"conv1_feature_maps_epoch_{epoch:02d}.png",
        dpi=150,
    )
    plt.close(figure1)

    figure2, axes2 = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes2.flat):
        axis.imshow(feature2[index].numpy(), cmap="gray")
        axis.set_title(f"ch {index}")
        axis.axis("off")
    figure2.suptitle(f"Conv2 feature maps - epoch {epoch}")
    figure2.tight_layout()
    figure2.savefig(
        OUTPUT_DIR / f"conv2_feature_maps_epoch_{epoch:02d}.png",
        dpi=150,
    )
    plt.close(figure2)


def visualize_conv1_filters(model: MnistCnn, epoch: int) -> None:
    """첫 convolution layer가 학습하는 3x3 filter weight를 그린다."""
    filters = model.conv1.weight.detach().cpu()[:, 0]
    figure, axes = plt.subplots(4, 4, figsize=(7, 7))

    for index, axis in enumerate(axes.flat):
        axis.imshow(filters[index].numpy(), cmap="coolwarm")
        axis.set_title(f"filter {index}")
        axis.axis("off")

    figure.suptitle(f"Conv1 filters - epoch {epoch}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"conv1_filters_epoch_{epoch:02d}.png", dpi=150)
    plt.close(figure)


def plot_training_curves(
    loss_history: list[float],
    accuracy_history: list[float],
) -> None:
    """CNN 학습 loss와 test accuracy를 저장한다."""
    epochs = range(1, len(loss_history) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(epochs, loss_history, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("MNIST CNN training loss")

    axes[1].plot(epochs, accuracy_history, marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("MNIST CNN test accuracy")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
