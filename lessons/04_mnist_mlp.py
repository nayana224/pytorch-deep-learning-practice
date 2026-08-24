from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


OUTPUT_DIR = Path("outputs/04_mnist_mlp")


class MnistMlp(nn.Module):
    """MNIST를 이용해 MLP, logits, softmax, cross-entropy를 복습한다."""

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


def print_softmax_checkpoint(
    model: MnistMlp,
    image: torch.Tensor,
    label: int,
    device: torch.device,
) -> None:
    """logits, softmax, cross-entropy의 연결을 한 sample로 확인한다."""
    model.eval()
    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))[0]
        probability = torch.softmax(logits, dim=0)
        manual_ce = -torch.log(probability[label])
        torch_ce = nn.functional.cross_entropy(
            logits.unsqueeze(0),
            torch.tensor([label], device=device),
        )

    print("\n=== Softmax / Cross-Entropy checkpoint ===")
    print(f"true label: {label}")
    print(f"predicted class: {probability.argmax().item()}")
    print(f"sum(softmax): {probability.sum().item():.6f}")
    print(f"manual -log(p_true): {manual_ce.item():.6f}")
    print(f"torch CrossEntropyLoss: {torch_ce.item():.6f}")


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device}")

    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)

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

        accuracy, _ = evaluate(model, test_loader, device)
        mean_loss = running_loss / len(train_loader)
        loss_history.append(mean_loss)
        accuracy_history.append(accuracy)
        print(f"epoch={epoch + 1} loss={mean_loss:.4f} test_accuracy={accuracy:.4f}")
        plot_first_layer_weights(model, epoch=epoch + 1)

    accuracy, confusion = evaluate(model, test_loader, device)
    print(f"final test accuracy: {accuracy:.4f}")
    sample_image, sample_label = test_dataset[0]
    print_softmax_checkpoint(model, sample_image, sample_label, device)

    plot_training_curves(loss_history, accuracy_history)
    plot_predictions(model, test_loader, device)
    plot_confusion_matrix(confusion)
    print(f"saved: {OUTPUT_DIR}")


def plot_training_curves(loss_history: list[float], accuracy_history: list[float]) -> None:
    epochs = range(1, len(loss_history) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))
    axes[0].plot(epochs, loss_history, marker="o")
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy loss", title="MNIST MLP training loss")
    axes[1].plot(epochs, accuracy_history, marker="o")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="MNIST MLP test accuracy", ylim=(0.0, 1.0))
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def plot_first_layer_weights(model: MnistMlp, epoch: int) -> None:
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


def plot_predictions(model: MnistMlp, loader: DataLoader, device: torch.device) -> None:
    images, labels = next(iter(loader))
    images_device = images[:16].to(device)
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(images_device), dim=1).cpu()

    predictions = probabilities.argmax(dim=1)
    confidence = probabilities.max(dim=1).values
    figure, axes = plt.subplots(4, 4, figsize=(8, 8))
    for index, axis in enumerate(axes.flat):
        axis.imshow(images[index, 0].numpy(), cmap="gray")
        axis.set_title(
            f"p={predictions[index].item()} ({confidence[index].item():.2f})\n"
            f"true={labels[index].item()}"
        )
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "test_predictions.png", dpi=150)
    plt.close(figure)


def plot_confusion_matrix(confusion: torch.Tensor) -> None:
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(confusion.numpy(), cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("MNIST confusion matrix")
    axis.set_xticks(range(10))
    axis.set_yticks(range(10))
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
