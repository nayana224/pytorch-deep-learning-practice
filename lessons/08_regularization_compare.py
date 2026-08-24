import torch
from torch import nn


class SmallClassifier(nn.Module):
    """Dropout 적용 여부를 쉽게 비교할 수 있는 작은 분류기."""

    def __init__(self, dropout_probability: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout_probability),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def main() -> None:
    torch.manual_seed(0)
    x = torch.randn(8, 20)

    model = SmallClassifier(dropout_probability=0.5)

    model.train()
    train_output_1 = model(x)
    train_output_2 = model(x)

    model.eval()
    with torch.no_grad():
        eval_output_1 = model(x)
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


if __name__ == "__main__":
    main()
