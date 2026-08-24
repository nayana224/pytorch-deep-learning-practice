from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/08_regularization_compare")


class Classifier(nn.Module):
    """작은 데이터에서 regularization 효과를 비교하는 MLP."""

    def __init__(self, dropout_probability: float = 0.0) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_probability),
            nn.Linear(128, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_dataset(samples: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """두 개의 noisy ring class를 만들어 과적합 비교에 사용한다."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, (samples,), generator=generator)
    angles = torch.rand(samples, generator=generator) * 2.0 * torch.pi
    radius = 1.0 + labels.float() * 0.8
    radius += torch.randn(samples, generator=generator) * 0.22

    x = torch.stack(
        [radius * torch.cos(angles), radius * torch.sin(angles)],
        dim=1,
    )
    return x, labels


def train(
    dropout_probability: float,
    weight_decay: float,
) -> tuple[Classifier, list[float], list[float]]:
    torch.manual_seed(0)
    train_x, train_y = make_dataset(samples=80, seed=0)
    test_x, test_y = make_dataset(samples=1000, seed=1)

    model = Classifier(dropout_probability=dropout_probability)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    train_accuracy_history: list[float] = []
    test_accuracy_history: list[float] = []

    for _ in range(400):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(train_x), train_y)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_accuracy = (model(train_x).argmax(dim=1) == train_y).float().mean().item()
            test_accuracy = (model(test_x).argmax(dim=1) == test_y).float().mean().item()
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)

    return model, train_accuracy_history, test_accuracy_history


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = {
        "No regularization": (0.0, 0.0),
        "L2 weight decay": (0.0, 1e-3),
        "Dropout": (0.5, 0.0),
    }

    results: dict[str, tuple[Classifier, list[float], list[float]]] = {}
    for name, (dropout, weight_decay) in experiments.items():
        result = train(dropout, weight_decay)
        results[name] = result
        _, train_history, test_history = result
        print(
            f"{name:17s} "
            f"train_acc={train_history[-1]:.3f} "
            f"test_acc={test_history[-1]:.3f} "
            f"gap={train_history[-1] - test_history[-1]:.3f}"
        )

    plot_accuracy_curves(results)
    plot_decision_boundaries(results)
    print("train-test gap이 줄어드는지 비교하세요.")
    print(f"saved: {OUTPUT_DIR}")


def plot_accuracy_curves(
    results: dict[str, tuple[Classifier, list[float], list[float]]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    for name, (_, train_history, test_history) in results.items():
        axes[0].plot(train_history, label=name)
        axes[1].plot(test_history, label=name)

    axes[0].set(xlabel="Epoch", ylabel="Accuracy", title="Training accuracy", ylim=(0.0, 1.02))
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Test accuracy", ylim=(0.0, 1.02))
    axes[0].legend()
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "accuracy_curves.png", dpi=150)
    plt.close(figure)


def plot_decision_boundaries(
    results: dict[str, tuple[Classifier, list[float], list[float]]],
) -> None:
    train_x, train_y = make_dataset(samples=80, seed=0)
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-2.5, 2.5, 200),
        torch.linspace(-2.5, 2.5, 200),
        indexing="xy",
    )
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (name, (model, _, _)) in zip(axes, results.items()):
        model.eval()
        with torch.no_grad():
            probability = torch.softmax(model(grid), dim=1)[:, 1].reshape(200, 200)
        axis.contourf(
            grid_x.numpy(),
            grid_y.numpy(),
            probability.numpy(),
            levels=20,
            cmap="coolwarm",
        )
        axis.scatter(
            train_x[:, 0].numpy(),
            train_x[:, 1].numpy(),
            c=train_y.numpy(),
            cmap="coolwarm",
            edgecolors="black",
            s=25,
        )
        axis.set_title(name)
        axis.set_xlabel("x1")
        axis.set_ylabel("x2")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "decision_boundaries.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
