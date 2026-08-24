import matplotlib.pyplot as plt
import torch
from torch import nn


class DeepMlp(nn.Module):
    """초기화 방식에 따른 activation 분포를 비교하기 위한 MLP."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(100, 100),
                nn.Linear(100, 100),
                nn.Linear(100, 100),
                nn.Linear(100, 100),
            ]
        )

    def forward_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations = []
        for layer in self.layers:
            x = torch.relu(layer(x))
            activations.append(x)
        return activations


def initialize(model: DeepMlp, mode: str) -> None:
    for layer in model.layers:
        if mode == "normal":
            nn.init.normal_(layer.weight, mean=0.0, std=1.0)
        elif mode == "xavier":
            nn.init.xavier_normal_(layer.weight)
        elif mode == "he":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        else:
            raise ValueError(f"Unknown initialization: {mode}")
        nn.init.zeros_(layer.bias)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(512, 100)

    modes = ["normal", "xavier", "he"]

    for mode in modes:
        model = DeepMlp()
        initialize(model, mode)
        activations = model.forward_activations(x)

        means = [a.mean().item() for a in activations]
        stds = [a.std().item() for a in activations]

        print(f"\n[{mode}]")
        for index, (mean, std) in enumerate(zip(means, stds), start=1):
            print(f"layer={index} mean={mean:.4f} std={std:.4f}")

        plt.plot(range(1, len(stds) + 1), stds, marker="o", label=mode)

    plt.xlabel("Layer")
    plt.ylabel("Activation std")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/initialization_compare.png", dpi=150)
    print("\nsaved: outputs/initialization_compare.png")


if __name__ == "__main__":
    main()
