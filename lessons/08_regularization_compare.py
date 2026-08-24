from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/08_regularization_compare")


class SmallClassifier(nn.Module):
    """Dropout 적용 여부를 쉽게 비교할 수 있는 작은 분류기."""

    def __init__(self, dropout_probability: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(20, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_probability)
        self.fc2 = nn.Linear(64, 2)

    def hidden_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Dropout 직전 hidden activation을 반환한다."""
        return self.relu(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden_activation(x)
        hidden = self.dropout(hidden)
        return self.fc2(hidden)


def main() -> None:
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x = torch.randn(128, 20)

    model = SmallClassifier(dropout_probability=0.5)
    hidden = model.hidden_activation(x).detach()

    model.train()
    dropped_1 = model.dropout(hidden).detach()
    dropped_2 = model.dropout(hidden).detach()
    train_output_1 = model.fc2(dropped_1)
    train_output_2 = model.fc2(dropped_2)

    model.eval()
    with torch.no_grad():
        eval_hidden = model.dropout(hidden)
        eval_output_1 = model.fc2(eval_hidden)
        eval_output_2 = model(x)

    train_difference = (train_output_1 - train_output_2).abs().mean().item()
    eval_difference = (eval_output_1 - eval_output_2).abs().mean().item()

    print(f"train mode difference: {train_difference:.6f}")
    print(f"eval mode difference:  {eval_difference:.6f}")
    print("\nDropout은 train mode에서만 확률적으로 동작한다.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    print(f"L2-style weight_decay: {optimizer.param_groups[0]['weight_decay']}")

    plot_dropout_masks(hidden, dropped_1, dropped_2)
    plot_activation_distributions(hidden, dropped_1)
    print(f"시각화 저장 위치: {OUTPUT_DIR}")
    plt.show()


def plot_dropout_masks(
    hidden: torch.Tensor,
    dropped_1: torch.Tensor,
    dropped_2: torch.Tensor,
) -> None:
    """같은 hidden activation에 서로 다른 dropout mask가 적용됨을 보여준다."""
    sample = 0
    values = [
        hidden[sample].numpy(),
        dropped_1[sample].numpy(),
        dropped_2[sample].numpy(),
    ]
    titles = ["Before dropout", "Dropout pass 1", "Dropout pass 2"]

    figure, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for axis, value, title in zip(axes, values, titles):
        axis.bar(range(len(value)), value)
        axis.set_ylabel("Activation")
        axis.set_title(title)

    axes[-1].set_xlabel("Hidden unit index")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "dropout_masks.png", dpi=150)


def plot_activation_distributions(
    hidden: torch.Tensor,
    dropped: torch.Tensor,
) -> None:
    """Dropout 전후 activation 분포와 zero 비율을 비교한다."""
    hidden_values = hidden.flatten().numpy()
    dropped_values = dropped.flatten().numpy()
    zero_ratio = (dropped == 0).float().mean().item()

    plt.figure(figsize=(8, 5))
    plt.hist(hidden_values, bins=40, alpha=0.6, label="Before dropout")
    plt.hist(dropped_values, bins=40, alpha=0.6, label="After dropout")
    plt.xlabel("Activation value")
    plt.ylabel("Count")
    plt.title(f"Activation distribution (zero ratio={zero_ratio:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "activation_distribution.png", dpi=150)


if __name__ == "__main__":
    main()
