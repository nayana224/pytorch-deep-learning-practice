from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/06_optimizer_compare")
OptimizerFactory = Callable[[list[nn.Parameter]], torch.optim.Optimizer]


class SmallRegressor(nn.Module):
    """Optimizer 비교에서 동일하게 사용하는 작은 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_problem() -> tuple[torch.Tensor, torch.Tensor]:
    """비선형 회귀 문제를 고정해 optimizer만 비교한다."""
    x = torch.linspace(-2.0, 2.0, 200).unsqueeze(1)
    y = 3.0 * x - 0.5 + 0.3 * torch.sin(5.0 * x)
    return x, y


def train(
    optimizer_factory: OptimizerFactory,
    epochs: int = 200,
) -> tuple[list[float], torch.Tensor]:
    """같은 초기 weight에서 optimizer 하나만 바꾸어 학습한다."""
    torch.manual_seed(0)
    x, y = make_problem()
    model = SmallRegressor()
    optimizer = optimizer_factory(list(model.parameters()))
    criterion = nn.MSELoss()
    history: list[float] = []

    for _ in range(epochs):
        optimizer.zero_grad()
        prediction = model(x)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    with torch.no_grad():
        final_prediction = model(x).squeeze(1)
    return history, final_prediction


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments: dict[str, OptimizerFactory] = {
        "SGD": lambda params: torch.optim.SGD(params, lr=0.01),
        "Momentum": lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9),
        "Adagrad": lambda params: torch.optim.Adagrad(params, lr=0.05),
        "RMSprop": lambda params: torch.optim.RMSprop(params, lr=0.01, alpha=0.99),
        "Adam": lambda params: torch.optim.Adam(params, lr=0.01),
    }

    histories: dict[str, list[float]] = {}
    predictions: dict[str, torch.Tensor] = {}

    for name, factory in experiments.items():
        history, prediction = train(factory)
        histories[name] = history
        predictions[name] = prediction
        print(f"{name:8s} final_loss={history[-1]:.6f}")

    plot_loss_curves(histories)
    plot_final_predictions(predictions)
    print(f"saved: {OUTPUT_DIR}")


def plot_loss_curves(histories: dict[str, list[float]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for name, history in histories.items():
        axis.semilogy(history, label=name)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("MSE loss (log scale)")
    axis.set_title("Optimizer comparison with identical initialization")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "loss_curves.png", dpi=150)
    plt.close(figure)


def plot_final_predictions(predictions: dict[str, torch.Tensor]) -> None:
    x, y = make_problem()
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.scatter(x.squeeze(1).numpy(), y.squeeze(1).numpy(), s=12, label="Training data")
    for name, prediction in predictions.items():
        axis.plot(x.squeeze(1).numpy(), prediction.numpy(), label=name)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_title("Final model fit by optimizer")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "final_predictions.png", dpi=150)
    plt.close(figure)


if __name__ == "__main__":
    main()
