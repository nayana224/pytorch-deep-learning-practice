from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

OUTPUT_DIR = Path("outputs/03_pixel_permutation")


class PermutedDataset(Dataset):
    """모든 이미지에 같은 pixel permutation을 적용한다."""

    def __init__(self, base_dataset: Dataset, permutation: torch.Tensor) -> None:
        self.base_dataset = base_dataset
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]
        flat = image.flatten()[self.permutation]
        return flat.reshape_as(image), label


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.features(x), 1))


def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int = 3,
) -> list[float]:
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    accuracy_history: list[float] = []

    for _ in range(epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images).argmax(1)
                correct += (predictions == labels).sum().item()
                total += labels.numel()
        accuracy_history.append(correct / total)

    return accuracy_history


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)

    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(28 * 28, generator=generator)
    permuted_train = PermutedDataset(train_set, permutation)
    permuted_test = PermutedDataset(test_set, permutation)

    original_image, label = test_set[0]
    permuted_image, _ = permuted_test[0]
    figure, axes = plt.subplots(1, 2, figsize=(7, 3))
    axes[0].imshow(original_image[0], cmap="gray")
    axes[0].set_title(f"Original, label={label}")
    axes[1].imshow(permuted_image[0], cmap="gray")
    axes[1].set_title("Same pixels, permuted positions")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "input_comparison.png", dpi=150)
    plt.close(figure)

    conditions = {
        "MLP original": (Mlp, train_set, test_set),
        "MLP permuted": (Mlp, permuted_train, permuted_test),
        "CNN original": (Cnn, train_set, test_set),
        "CNN permuted": (Cnn, permuted_train, permuted_test),
    }

    histories: dict[str, list[float]] = {}
    for name, (model_class, current_train, current_test) in conditions.items():
        # 같은 architecture끼리는 같은 random seed에서 시작해 입력 조건만 비교한다.
        torch.manual_seed(0)
        model = model_class()
        train_loader = DataLoader(
            current_train,
            batch_size=128,
            shuffle=True,
            num_workers=2,
        )
        test_loader = DataLoader(
            current_test,
            batch_size=256,
            shuffle=False,
            num_workers=2,
        )
        histories[name] = train_and_evaluate(model, train_loader, test_loader, device)
        print(f"{name}: final_accuracy={histories[name][-1]:.4f}")

    figure = plt.figure(figsize=(9, 5))
    for name, history in histories.items():
        plt.plot(range(1, len(history) + 1), history, marker="o", label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy")
    plt.ylim(0.0, 1.0)
    plt.title("What happens when MNIST spatial structure is destroyed?")
    plt.legend()
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(figure)

    print(f"visualizations: {OUTPUT_DIR}")
    print("핵심 질문: 같은 pixel 값이어도 공간적 이웃 관계를 없애면 MLP와 CNN은 어떻게 달라지는가?")


if __name__ == "__main__":
    main()
