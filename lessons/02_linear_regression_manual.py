from pathlib import Path

import matplotlib.pyplot as plt
import torch


OUTPUT_DIR = Path("outputs/02_linear_regression_manual")


def main() -> None:
    """Autograd 없이 선형 회귀의 gradient descent를 직접 구현한다."""
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0

    w = torch.tensor(0.0)
    b = torch.tensor(0.0)

    learning_rate = 0.05
    epochs = 200
    n = x.numel()

    loss_history: list[float] = []
    weight_history: list[float] = []
    bias_history: list[float] = []
    snapshots: dict[int, torch.Tensor] = {}
    snapshot_epochs = {0, 20, 50, 100, epochs - 1}

    for epoch in range(epochs):
        y_hat = w * x + b
        error = y_hat - y
        loss = 0.5 * torch.mean(error**2)

        # C = (1 / 2N) sum((y_hat - y)^2)의 해석적 gradient다.
        grad_w = torch.sum(error * x) / n
        grad_b = torch.sum(error) / n

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        loss_history.append(loss.item())
        weight_history.append(w.item())
        bias_history.append(b.item())

        if epoch in snapshot_epochs:
            snapshots[epoch] = (w * x + b).detach().clone()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f} "
                f"w={w.item():.4f} "
                f"b={b.item():.4f}"
            )

    plot_regression_progress(x, y, snapshots)
    plot_training_history(loss_history, weight_history, bias_history)
    print(f"\n시각화 저장 위치: {OUTPUT_DIR}")
    print("Expected: w ~= 2, b ~= 1")
    plt.show()


def plot_regression_progress(
    x: torch.Tensor,
    y: torch.Tensor,
    snapshots: dict[int, torch.Tensor],
) -> None:
    """학습이 진행되면서 회귀 직선이 정답에 접근하는 모습을 그린다."""
    plt.figure(figsize=(8, 5))
    plt.scatter(x.numpy(), y.numpy(), label="Training data")

    for epoch, prediction in snapshots.items():
        plt.plot(x.numpy(), prediction.numpy(), label=f"Epoch {epoch}")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Manual gradient descent: regression progress")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regression_progress.png", dpi=150)


def plot_training_history(
    loss_history: list[float],
    weight_history: list[float],
    bias_history: list[float],
) -> None:
    """Loss와 학습 parameter의 변화를 함께 확인한다."""
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(loss_history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss history")
    axes[0].set_yscale("log")

    axes[1].plot(weight_history, label="w")
    axes[1].plot(bias_history, label="b")
    axes[1].axhline(2.0, linestyle="--", label="target w=2")
    axes[1].axhline(1.0, linestyle="--", label="target b=1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Parameter value")
    axes[1].set_title("Parameter convergence")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history.png", dpi=150)


if __name__ == "__main__":
    main()
