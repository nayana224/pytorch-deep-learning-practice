import torch
from torch import nn


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

    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = 2.0 * x + 1.0

    model = LinearRegression()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    for epoch in range(200):
        optimizer.zero_grad()

        y_hat = model(x)
        loss = criterion(y_hat, y)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 199:
            weight = model.linear.weight.item()
            bias = model.linear.bias.item()
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f} "
                f"w={weight:.4f} "
                f"b={bias:.4f}"
            )

    print("\n학습 loop 순서:")
    print("zero_grad -> forward -> loss -> backward -> optimizer.step")


if __name__ == "__main__":
    main()
