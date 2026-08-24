from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/03_linear_regression_torch")


class LinearRegression(nn.Module):
    """한 개의 입력으로 연속값을 예측하는 선형 회귀 모델."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def main() -> None:
    """PyTorch 표준 학습 loop로 선형 회귀를 학습한다."""
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * x + 1.0

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    loss_history: list[float] = []
    weight_history: list[float] = []
    bias_history: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    snapshot_epochs = {0, 20, 50, 100, 199}

    for epoch in range(200):
        optimizer.zero_grad()

        y_hat = model(x)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

        weight = model.linear.weight.item()
        bias = model.linear.bias.item()
        loss_history.append(loss.item())
        weight_history.append(weight)
        bias_history.append(bias)

        if epoch in snapshot_epochs:
            with torch.no_grad():
                snapshots[epoch] = model(x).squeeze(1).clone()

        if epoch % 20 == 0 or epoch == 199:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f} "
                f"w={weight:.4f} "
                f"b={bias:.4f}"
            )

    plot_results(x.squeeze(1), y.squeeze(1), snapshots)
    plot_history(loss_history, weight_history, bias_history)

    print("\n학습 loop 순서:")
    print("zero_grad -> forward -> loss -> backward -> optimizer.step")
    print(f"시각화 저장 위치: {OUTPUT_DIR}")
    plt.show()


def plot_results(
    x: torch.Tensor,
    y: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
) -> None:
    """PyTorch가 학습한 회귀 직선의 변화를 시각화한다."""
    plt.figure(figsize=(8, 5))
    plt.scatter(x.numpy(), y.numpy(), label="Training data")

    for epoch, prediction in snapshots.items():
        plt.plot(x.numpy(), prediction.numpy(), label=f"Epoch {epoch}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("PyTorch linear regression progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regression_progress.png", dpi=150)


def plot_history(
    loss_history: list[float],
    weight_history: list[float],
    bias_history: list[float],
) -> None:
    """Loss와 nn.Linear 내부 weight, bias의 변화를 그린다."""
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(loss_history)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].set_title("Training loss")

    axes[1].plot(weight_history, label="weight")
    axes[1].plot(bias_history, label="bias")
    axes[1].axhline(2.0, linestyle="--", label="target weight=2")
    axes[1].axhline(1.0, linestyle="--", label="target bias=1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Parameter value")
    axes[1].set_title("nn.Linear parameter convergence")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history.png", dpi=150)


if __name__ == "__main__":
    main()
