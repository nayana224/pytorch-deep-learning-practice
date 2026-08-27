from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


OUTPUT_DIR = Path("outputs/00_inspect_mnist")


def compute_dataset_statistics(loader: DataLoader) -> tuple[float, float]:
    """MNIST 전체 training set의 pixel 평균과 표준편차를 계산한다."""
    pixel_sum = 0.0
    pixel_squared_sum = 0.0
    pixel_count = 0

    for images, _ in loader:
        pixel_sum += images.sum().item()
        pixel_squared_sum += (images**2).sum().item()
        pixel_count += images.numel()

    mean = pixel_sum / pixel_count
    variance = pixel_squared_sum / pixel_count - mean**2
    return mean, variance**0.5


def plot_samples_by_class(dataset: datasets.MNIST) -> None:
    """각 class에서 네 장씩 골라 실제 입력 이미지를 확인한다."""
    figure, axes = plt.subplots(4, 10, figsize=(14, 6))

    for class_index in range(10):
        indices = (dataset.targets == class_index).nonzero(as_tuple=True)[0][:4]
        for row, index in enumerate(indices):
            image, label = dataset[index.item()]
            axes[row, class_index].imshow(image[0].numpy(), cmap="gray")
            axes[row, class_index].axis("off")
            if row == 0:
                axes[row, class_index].set_title(f"class {label}")

    figure.suptitle("MNIST samples by class")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "samples_by_class.png", dpi=150)
    plt.close(figure)


def plot_class_distribution(dataset: datasets.MNIST) -> None:
    """Training set의 class별 sample 수를 확인한다."""
    counts = torch.bincount(dataset.targets, minlength=10)

    figure = plt.figure(figsize=(8, 5))
    plt.bar(range(10), counts.numpy())
    plt.xticks(range(10))
    plt.xlabel("Class")
    plt.ylabel("Number of samples")
    plt.title("MNIST training class distribution")
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_distribution.png", dpi=150)
    plt.close(figure)


def plot_class_mean_images(dataset: datasets.MNIST) -> None:
    """같은 숫자 class의 평균 이미지를 만들어 공통 형태를 관찰한다."""
    images = dataset.data.float() / 255.0
    targets = dataset.targets

    figure, axes = plt.subplots(2, 5, figsize=(10, 4))
    for class_index, axis in enumerate(axes.flat):
        mean_image = images[targets == class_index].mean(dim=0)
        axis.imshow(mean_image.numpy(), cmap="gray", vmin=0.0, vmax=1.0)
        axis.set_title(f"class {class_index}")
        axis.axis("off")

    figure.suptitle("Mean image for each MNIST class")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "class_mean_images.png", dpi=150)
    plt.close(figure)


def plot_pixel_histogram(dataset: datasets.MNIST) -> None:
    """MNIST pixel intensity가 어떤 값에 많이 분포하는지 확인한다."""
    values = dataset.data[:5000].float().flatten() / 255.0

    figure = plt.figure(figsize=(8, 5))
    plt.hist(values.numpy(), bins=50)
    plt.xlabel("Pixel intensity")
    plt.ylabel("Count")
    plt.title("MNIST pixel intensity distribution (first 5,000 samples)")
    plt.yscale("log")
    plt.tight_layout()
    figure.savefig(OUTPUT_DIR / "pixel_histogram.png", dpi=150)
    plt.close(figure)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

    image, label = train_dataset[0]
    print("=== Dataset ===")
    print(f"train samples: {len(train_dataset)}")
    print(f"test samples:  {len(test_dataset)}")
    print("\n=== One sample ===")
    print(f"image shape [C, H, W]: {tuple(image.shape)}")
    print(f"dtype: {image.dtype}")
    print(f"pixel range: [{image.min().item():.3f}, {image.max().item():.3f}]")
    print(f"label: {label}")

    loader = DataLoader(train_dataset, batch_size=1024, shuffle=False)
    mean, std = compute_dataset_statistics(loader)
    counts = torch.bincount(train_dataset.targets, minlength=10)

    print("\n=== Training-set statistics ===")
    print(f"pixel mean: {mean:.6f}")
    print(f"pixel std:  {std:.6f}")
    print(f"class counts: {counts.tolist()}")

    plot_samples_by_class(train_dataset)
    plot_class_distribution(train_dataset)
    plot_class_mean_images(train_dataset)
    plot_pixel_histogram(train_dataset)

    print(f"\nvisualizations: {OUTPUT_DIR}")
    print("먼저 samples_by_class.png와 class_mean_images.png를 비교하세요.")


if __name__ == "__main__":
    main()
