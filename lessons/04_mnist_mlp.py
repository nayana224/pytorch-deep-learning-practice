import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistMlp(nn.Module):
    """28x28 MNIST 영상을 10개 숫자 class로 분류한다."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


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

    for epoch in range(3):
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
        print(
            f"epoch={epoch + 1} "
            f"loss={mean_loss:.4f} "
            f"test_accuracy={accuracy:.4f}"
        )


if __name__ == "__main__":
    main()
