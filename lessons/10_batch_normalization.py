from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/10_batch_normalization")


class DeepMlp(nn.Module):
    """Batch Normalization 적용 여부에 따른 activation 변화를 비교한다."""

    def __init__(self, use_batch_norm: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(5):
            block: list[nn.Module] = [nn.Linear(64, 64)]
            if use_batch_norm:
                block.append(nn.BatchNorm1d(64))
            block.append(nn.ReLU())
            self.layers.append(nn.Sequential(*block))

    def forward_activations(self, x: torch.Tensor) -> list[torch.Tensor]:
        activations: list[torch.Tensor] = []
        for layer in self.layers:
            x = layer(x)
            activations.append(x)
        return activations


def collect_stats(activations: list[torch.Tensor]) -> tuple[list[float], list[float]]:
    means = [activation.mean().item() for activation in activations]
    stds = [activation.std().item() for activation in activations]
    return means, stds


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    x = torch.randn(1024, 64) * 5.0 + 3.0
    plain_model = DeepMlp(use_batch_norm=False)
    bn_model = DeepMlp(use_batch_norm=True)

    plain_model.train()
    bn_model.train()

    plain_activations = plain_model.forward_activations(x)
    bn_activations = bn_model.forward_activations(x)

    plain_means, plain_stds = collect_stats(plain_activations)
    bn_means, bn_stds = collect_stats(bn_activations)

    layers = range(1, len(plain_means) + 1)
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(layers, plain_means, marker="o", label="Without BN")
    axes[0].plot(layers, bn_means, marker="o", label="With BN")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Activation mean")
    axes[0].set_title("Activation mean across layers")
    axes[0].legend()

    axes[1].plot(layers, plain_stds, marker="o", label="Without BN")
    axes[1].plot(layers, bn_stds, marker="o", label="With BN")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Activation std")
    axes[1].set_title("Activation standard deviation across layers")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "activation_statistics.png", dpi=150)
    plt.close(figure)

    first_bn = bn_model.layers[0][1]
    if isinstance(first_bn, nn.BatchNorm1d):
        print(f"running_mean shape: {tuple(first_bn.running_mean.shape)}")
        print(f"running_var shape:  {tuple(first_bn.running_var.shape)}")
        print(f"learnable gamma shape: {tuple(first_bn.weight.shape)}")
        print(f"learnable beta shape:  {tuple(first_bn.bias.shape)}")

    print("BatchNorm은 mini-batch 통계로 activation scale을 정규화한 뒤")
    print("learnable gamma와 beta로 다시 scale/shift한다.")
    print(f"saved: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
