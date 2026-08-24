from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


OUTPUT_DIR = Path("outputs/04_mnist_mlp")


class MnistMlp(nn.Module):
    """28x28 MNIST 영상을 10개 숫자 class로 분류한다."""

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
) -> float:
    """전체 데이터에서 classification accuracy를 계산한다."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    loss_history: list[float] = []
    accuracy_history: list[float] = []

    plot_first_layer_weights(model, epoch=0)

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

        accuracy = evaluate(model, test_loader, device)
        mean_loss = running_loss / len(train_loader)
        loss_history.append(mean_loss)
        accuracy_history.append(accuracy)

        print(
            f"epoch={epoch + 1} "
            f"loss={mean_loss:.4f} "
            f"test_accuracy={accuracy:.4f}"
        )
        plot_first_layer_weights(model, epoch=epoch + 1)

    plot_training_curves(loss_history, accuracy_history)
    plot_predictions(model, test_loader, device)

    print(f"시각화 저장 위치: {OUTPUT_DIR}")
    plt.show()


def plot_training_curves(
    loss_history: list[float],
    accuracy_history: list[float],
) -> None:
    """Epoch에 따른 loss와 test accuracy를 시각화한다."""
    epochs = range(1, len(loss_history) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(epochs, loss_history, marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("MNIST MLP training loss")

    axes[1].plot(epochs, accuracy_history, marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("MNIST MLP test accuracy")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)


def plot_first_layer_weights(model: MnistMlp, epoch: int) -> None:
    """첫 Linear layer의 weight를 28x28 이미지처럼 관찰한다."""
    weights = model.fc1.weight.detach().cpu().reshape(128, 28, 28)
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))

    for index, axis in enumerate(axes.flat):
        axis.imshow(weights[index].numpy(), cmap="coolwarm")
        axis.set_title(f"Neuron {index}")
        axis.axis("off")

    figure.suptitle(f"First-layer weights after epoch {epoch}")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"first_layer_weights_epoch_{epoch:02d}.png", dpi=150)
    plt.close(figure)


def plot_predictions(
    model: MnistMlp,
    loader: DataLoader,
    device: torch.device,
) -> None:
    """고정된 test sample의 prediction을 이미지와 함께 확인한다."""
    images, labels = next(iter(loader))
    images = images[:16].to(device)
    labels = labels[:16]

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


if __name__ == "__main__":
    main()
