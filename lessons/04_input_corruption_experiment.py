from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

OUTPUT_DIR = Path("outputs/04_input_corruption")


class CorruptedDataset(Dataset):
    """MNIST test image에 고정된 입력 손상을 적용한다."""

    def __init__(self, base_dataset: Dataset, mode: str) -> None:
        self.base_dataset = base_dataset
        self.mode = mode

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label = self.base_dataset[index]
        image = image.clone()

        if self.mode == "noise":
            generator = torch.Generator().manual_seed(index)
            noise = torch.randn(image.shape, generator=generator) * 0.30
            image = torch.clamp(image + noise, 0.0, 1.0)
        elif self.mode == "occlusion":
            image[:, 9:19, 9:19] = 0.0
        else:
            raise ValueError(f"unknown corruption mode: {self.mode}")

        return image, label


class Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Cnn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.features(x), 1))


def train(model: nn.Module, loader: DataLoader, device: torch.device) -> None:
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for _ in range(5):
        model.train()
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.ToTensor()
    train_set = datasets.MNIST("./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=transform)
    noise_set = CorruptedDataset(test_set, "noise")
    occlusion_set = CorruptedDataset(test_set, "occlusion")

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loaders = {
        "clean": DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2),
        "noise": DataLoader(noise_set, batch_size=256, shuffle=False, num_workers=2),
        "occlusion": DataLoader(occlusion_set, batch_size=256, shuffle=False, num_workers=2),
    }

    clean_image, label = test_set[0]
    noise_image, _ = noise_set[0]
    occluded_image, _ = occlusion_set[0]
    figure, axes = plt.subplots(1, 3, figsize=(9, 3))
    for axis, image, title in zip(
        axes,
        (clean_image, noise_image, occluded_image),
        (f"Clean, label={label}", "Gaussian noise", "Center occlusion"),
    ):
        axis.imshow(image[0], cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "corruption_examples.png", dpi=150)
    plt.close(figure)

    results: dict[str, dict[str, float]] = {}
    for model_name, model in (("MLP", Mlp()), ("CNN", Cnn())):
        torch.manual_seed(0)
        train(model, train_loader, device)
        results[model_name] = {
            condition: evaluate(model, loader, device)
            for condition, loader in test_loaders.items()
        }
        print(f"{model_name}: {results[model_name]}")

    labels = ["clean", "noise", "occlusion"]
    x = torch.arange(len(labels), dtype=torch.float32)
    width = 0.35
    figure = plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, [results["MLP"][key] for key in labels], width, label="MLP")
    plt.bar(x + width / 2, [results["CNN"][key] for key in labels], width, label="CNN")
    plt.xticks(x.numpy(), labels)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Test accuracy")
    plt.title("Robustness to MNIST input corruption")
    plt.legend()
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(figure)

    print(f"visualizations: {OUTPUT_DIR}")
    print("핵심 질문: 입력이 깨졌을 때 MLP와 CNN의 성능은 어떤 방식으로 무너지는가?")


if __name__ == "__main__":
    main()
