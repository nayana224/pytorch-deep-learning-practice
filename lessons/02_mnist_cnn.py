from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

OUTPUT_DIR = Path("outputs/02_mnist_cnn")


class MnistCnn(nn.Module):
    """MNIST의 2차원 공간 구조를 convolution으로 학습하는 CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.relu(self.conv1(x))
        f2 = self.relu(self.conv2(self.pool(f1)))
        return f1, f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, f2 = self.features(x)
        x = self.pool(f2)
        return self.fc(torch.flatten(x, 1))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
    return correct / total


def save_feature_maps(model: MnistCnn, image: torch.Tensor, name: str) -> None:
    """같은 입력에 대한 feature map을 학습 전후로 저장한다."""
    was_training = model.training
    model.eval()
    with torch.no_grad():
        f1, f2 = model.features(image)
    model.train(was_training)

    for layer_name, feature in (("conv1", f1[0]), ("conv2", f2[0, :16])):
        figure, axes = plt.subplots(4, 4, figsize=(8, 8))
        for index, axis in enumerate(axes.flat):
            axis.imshow(feature[index].cpu().numpy(), cmap="gray")
            axis.set_title(f"ch {index}")
            axis.axis("off")
        figure.suptitle(f"{layer_name} feature maps - {name}")
        figure.tight_layout()
        figure.savefig(OUTPUT_DIR / f"{layer_name}_{name}.png", dpi=150)
        plt.close(figure)


def save_filters(model: MnistCnn, name: str) -> None:
    """실제로 학습되는 첫 convolution filter weight를 저장한다."""
    filters = model.conv1.weight.detach().cpu()[:, 0]
    figure, axes = plt.subplots(4, 4, figsize=(7, 7))
    for index, axis in enumerate(axes.flat):
        axis.imshow(filters[index].numpy(), cmap="coolwarm")
        axis.set_title(f"filter {index}")
        axis.axis("off")
    figure.suptitle(f"Conv1 filters - {name}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"filters_{name}.png", dpi=150)
    plt.close(figure)


def save_curves(losses: list[float], accuracies: list[float]) -> None:
    epochs = range(1, len(losses) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_title("CNN training loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[1].plot(epochs, accuracies, marker="o")
    axes[1].set_title("CNN test accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

    model = MnistCnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    fixed_image, fixed_label = test_set[0]
    fixed_image = fixed_image.unsqueeze(0).to(device)
    with torch.no_grad():
        f1, f2 = model.features(fixed_image)

    print(f"device: {device}")
    print(f"fixed sample label: {fixed_label}")
    print(f"input: {tuple(fixed_image.shape)}")
    print(f"conv1 feature map: {tuple(f1.shape)}")
    print(f"conv2 feature map: {tuple(f2.shape)}")
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    save_filters(model, "before_training")
    save_feature_maps(model, fixed_image, "before_training")

    losses: list[float] = []
    accuracies: list[float] = []
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        mean_loss = running_loss / len(train_loader)
        accuracy = evaluate(model, test_loader, device)
        losses.append(mean_loss)
        accuracies.append(accuracy)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f} test_accuracy={accuracy:.4f}")

    save_filters(model, "after_training")
    save_feature_maps(model, fixed_image, "after_training")
    save_curves(losses, accuracies)
    print(f"visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
