from collections.abc import Callable

import matplotlib.pyplot as plt
import torch
from torch import nn


OptimizerFactory = Callable[
    [list[nn.Parameter]],
    torch.optim.Optimizer,
]


def make_problem() -> tuple[torch.Tensor, torch.Tensor]:
    """동일한 optimizer 비교를 위해 고정된 회귀 문제를 만든다."""
    x = torch.linspace(-2.0, 2.0, 200).unsqueeze(1)
    y = 3.0 * x - 0.5 + 0.3 * torch.sin(5.0 * x)
    return x, y


def train(
    optimizer_factory: OptimizerFactory,
    epochs: int = 200,
) -> list[float]:
    torch.manual_seed(0)
    x, y = make_problem()

    model = nn.Sequential(
        nn.Linear(1, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
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

    return history


def main() -> None:
    experiments: dict[str, OptimizerFactory] = {
        "SGD": lambda params: torch.optim.SGD(params, lr=0.01),
        "Momentum": lambda params: torch.optim.SGD(
            params,
            lr=0.01,
            momentum=0.9,
        ),
        "Adam": lambda params: torch.optim.Adam(params, lr=0.01),
    }

    for name, factory in experiments.items():
        history = train(factory)
        plt.plot(history, label=name)
        print(f"{name}: final_loss={history[-1]:.6f}")

    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/optimizer_compare.png", dpi=150)
    print("saved: outputs/optimizer_compare.png")


if __name__ == "__main__":
    main()
