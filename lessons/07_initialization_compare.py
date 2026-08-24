from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/07_initialization_compare")


class DeepMlp(nn.Module):
    """Initialization과 activation에 따른 signal/gradient 흐름을 관찰한다."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(8)])
        self.activation = activation

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(x)
        if self.activation == "tanh":
            return torch.tanh(x)
        if self.activation == "relu":
            return torch.relu(x)
        raise ValueError(f"Unknown activation: {self.activation}")

    def forward_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations: list[torch.Tensor] = []
        for layer in self.layers:
            x = self.activate(layer(x))
            x.retain_grad()
            activations.append(x)
        return activations


def initialize(model: DeepMlp, mode: str) -> None:
    """강의에서 비교하는 대표 initialization을 적용한다."""
    for layer in model.layers:
        if mode == "small_normal":
            nn.init.normal_(layer.weight, mean=0.0, std=0.01)
        elif mode == "large_normal":
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        elif mode == "xavier":
            nn.init.xavier_normal_(layer.weight)
        elif mode == "he":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown initialization: {mode}")
        nn.init.zeros_(layer.bias)


def run_experiment(
    activation: str,
    initialization: str,
) -> tuple[list[float], list[float]]:
    torch.manual_seed(0)
    x = torch.randn(512, 100)
    model = DeepMlp(activation=activation)
    initialize(model, initialization)

    activations = model.forward_activations(x)
    loss = activations[-1].pow(2).mean()
    loss.backward()

    activation_stds = [value.detach().std().item() for value in activations]
    gradient_norms = [value.grad.norm().item() for value in activations]
    return activation_stds, gradient_norms


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    experiments = {
        "Sigmoid + small normal": ("sigmoid", "small_normal"),
        "Sigmoid + Xavier": ("sigmoid", "xavier"),
        "ReLU + large normal": ("relu", "large_normal"),
        "ReLU + He": ("relu", "he"),
    }

    activation_stats: dict[str, list[float]] = {}
    gradient_stats: dict[str, list[float]] = {}

    for name, (activation, initialization) in experiments.items():
        activation_stds, gradient_norms = run_experiment(activation, initialization)
        activation_stats[name] = activation_stds
        gradient_stats[name] = gradient_norms
        print(f"\n[{name}]")
        for layer, (std, grad) in enumerate(zip(activation_stds, gradient_norms), start=1):
            print(f"layer={layer} activation_std={std:.6f} gradient_norm={grad:.6e}")

    plot_statistics(activation_stats, gradient_stats)
    print("앞쪽 layer의 gradient가 지나치게 작아지는지 확인하세요.")
    print(f"saved: {OUTPUT_DIR}")


def plot_statistics(
    activation_stats: dict[str, list[float]],
    gradient_stats: dict[str, list[float]],
) -> None:
    layers = range(1, 9)

    figure1, axis1 = plt.subplots(figsize=(8, 5))
    for name, values in activation_stats.items():
        axis1.plot(layers, values, marker="o", label=name)
    axis1.set_xlabel("Layer")
    axis1.set_ylabel("Activation standard deviation")
    axis1.set_title("Forward signal propagation")
    axis1.legend()
    figure1.tight_layout()
    figure1.savefig(OUTPUT_DIR / "activation_std.png", dpi=150)
    plt.close(figure1)

    figure2, axis2 = plt.subplots(figsize=(8, 5))
    for name, values in gradient_stats.items():
        axis2.semilogy(layers, values, marker="o", label=name)
    axis2.set_xlabel("Layer")
    axis2.set_ylabel("Activation gradient norm")
    axis2.set_title("Backward gradient flow")
    axis2.legend()
    figure2.tight_layout()
    figure2.savefig(OUTPUT_DIR / "gradient_flow.png", dpi=150)
    plt.close(figure2)


if __name__ == "__main__":
    main()
