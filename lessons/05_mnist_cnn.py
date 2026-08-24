import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class MnistCnn(nn.Module):
    """Convolution과 pooling으로 MNIST를 분류하는 작은 CNN."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    torch.manual_seed(0)
    device = get_device()
    print(f"device: {device}")

    train_dataset = datasets.MNIST(
        "./data",
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=2,
    )

    model = MnistCnn().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 첫 batch를 통과시키며 shape 변화를 확인한다.
    images, _ = next(iter(train_loader))
    print(f"input: {tuple(images.shape)}")
    with torch.no_grad():
        feature_map = model.features(images.to(device))
    print(f"after features: {tuple(feature_map.shape)}")

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

        print(
            f"epoch={epoch + 1} "
            f"loss={running_loss / len(train_loader):.4f}"
        )


if __name__ == "__main__":
    main()
