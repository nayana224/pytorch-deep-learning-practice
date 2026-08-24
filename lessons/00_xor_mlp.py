from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/00_xor_mlp")


class SingleLayer(nn.Module):
    """XOR의 선형 분리 한계를 확인하기 위한 단일 선형 분류기."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


class XorMlp(nn.Module):
    """비선형 hidden layer로 XOR을 학습하는 작은 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 4),
            nn.Tanh(),
            nn.Linear(4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(1)


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> list[float]:
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    history: list[float] = []

    for _ in range(1000):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        history.append(loss.item())

    return history


def plot_decision_surface(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    title: str,
    filename: str,
) -> None:
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(-0.5, 1.5, 200),
        torch.linspace(-0.5, 1.5, 200),
        indexing="xy",
    )
    grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)

    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(grid)).reshape(200, 200)

    figure, axis = plt.subplots(figsize=(6, 5))
    contour = axis.contourf(
        grid_x.numpy(),
        grid_y.numpy(),
        probability.numpy(),
        levels=20,
        cmap="coolwarm",
    )
    figure.colorbar(contour, ax=axis, label="P(y=1)")
    axis.scatter(x[:, 0].numpy(), x[:, 1].numpy(), c=y.numpy(), cmap="coolwarm", edgecolors="black", s=100)
    axis.set_xlabel("x1")
    axis.set_ylabel("x2")
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    )
    y = torch.tensor([0.0, 1.0, 1.0, 0.0])

    linear_model = SingleLayer()
    mlp_model = XorMlp()

    linear_history = train(linear_model, x, y)
    mlp_history = train(mlp_model, x, y)

    print(f"single-layer final loss: {linear_history[-1]:.6f}")
    print(f"MLP final loss:          {mlp_history[-1]:.6f}")

    with torch.no_grad():
        linear_pred = (torch.sigmoid(linear_model(x)) >= 0.5).int()
        mlp_pred = (torch.sigmoid(mlp_model(x)) >= 0.5).int()
    print(f"single-layer prediction: {linear_pred.tolist()}")
    print(f"MLP prediction:          {mlp_pred.tolist()}")

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.semilogy(linear_history, label="Single linear layer")
    axis.semilogy(mlp_history, label="MLP + Tanh")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("BCE loss")
    axis.set_title("XOR learning curve")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    plt.close(figure)

    plot_decision_surface(
        linear_model,
        x,
        y,
        "Single layer: XOR is not linearly separable",
        "single_layer_surface.png",
    )
    plot_decision_surface(
        mlp_model,
        x,
        y,
        "MLP: nonlinear hidden representation solves XOR",
        "mlp_surface.png",
    )
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
