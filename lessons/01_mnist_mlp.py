from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


OUTPUT_DIR = Path("outputs/01_mnist_mlp")


class MnistMlp(nn.Module):
    """28x28 MNIST 이미지를 flatten한 뒤 fully connected layer로 분류한다."""

    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, torch.Tensor]:
    """Test accuracy와 confusion matrix를 계산한다."""
    model.eval()
    correct = 0
    total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.numel()
            for true_label, predicted_label in zip(labels.cpu(), predictions.cpu()):
                confusion[true_label, predicted_label] += 1

    return correct / total, confusion


def plot_training_curves(losses: list[float], accuracies: list[float]) -> None:
    """Epoch별 loss와 test accuracy를 저장한다."""
    epochs = range(1, len(losses) + 1)

    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].plot(epochs, losses, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("MLP training loss")

    axes[1].plot(epochs, accuracies, marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("MLP test accuracy")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def plot_first_layer_weights(model: MnistMlp) -> None:
    """첫 hidden neuron들이 784개 pixel에 주는 weight를 28x28로 다시 본다."""
    weights = model.fc1.weight.detach().cpu().reshape(128, 28, 28)
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))

    for index, axis in enumerate(axes.flat):
        axis.imshow(weights[index].numpy(), cmap="coolwarm")
        axis.set_title(f"hidden {index}")
        axis.axis("off")

    figure.suptitle("MLP first-layer weights after training")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "first_layer_weights.png", dpi=150)
    plt.close(figure)


def plot_confusion_matrix(confusion: torch.Tensor) -> None:
    """어떤 숫자끼리 자주 혼동하는지 confusion matrix로 확인한다."""
    figure = plt.figure(figsize=(7, 6))
    plt.imshow(confusion.numpy(), cmap="Blues")
    plt.colorbar()
    plt.xticks(range(10))
    plt.yticks(range(10))
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("MLP confusion matrix")
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


def plot_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """실제 test image와 예측 결과를 같이 저장한다."""
    images, labels = next(iter(loader))
    images = images[:16].to(device)

    model.eval()
    with torch.no_grad():
        predictions = model(images).argmax(dim=1).cpu()

    images = images.cpu()
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index, 0].numpy(), cmap="gray")
        axis.set_title(
            f"pred={predictions[index].item()}, true={labels[index].item()}"
        )
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()

    transform = transforms.ToTensor()
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

    model = MnistMlp().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    first_images, first_labels = next(iter(train_loader))
    print(f"device: {device}")
    print(f"input batch shape [B, C, H, W]: {tuple(first_images.shape)}")
    print(f"label batch shape [B]: {tuple(first_labels.shape)}")
    print(f"flattened features per image: {28 * 28}")
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    losses: list[float] = []
    accuracies: list[float] = []

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
        accuracy, confusion = evaluate(model, test_loader, device)
        losses.append(mean_loss)
        accuracies.append(accuracy)
        print(
            f"epoch={epoch + 1} "
            f"loss={mean_loss:.4f} "
            f"test_accuracy={accuracy:.4f}"
        )

    plot_training_curves(losses, accuracies)
    plot_first_layer_weights(model)
    plot_confusion_matrix(confusion)
    plot_predictions(model, test_loader, device)
    print(f"visualizations: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
