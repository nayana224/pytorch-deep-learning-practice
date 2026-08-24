import torch


def main() -> None:
    """Autograd 없이 선형 회귀의 gradient descent를 직접 구현한다."""
    torch.manual_seed(0)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    y = 2.0 * x + 1.0

    w = torch.tensor(0.0)
    b = torch.tensor(0.0)

    learning_rate = 0.05
    epochs = 200
    n = x.numel()

    for epoch in range(epochs):
        y_hat = w * x + b
        error = y_hat - y
        loss = 0.5 * torch.mean(error**2)

        # C = (1 / 2N) sum((y_hat - y)^2)의 해석적 gradient다.
        grad_w = torch.sum(error * x) / n
        grad_b = torch.sum(error) / n

        w = w - learning_rate * grad_w
        b = b - learning_rate * grad_b

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.6f} "
                f"w={w.item():.4f} "
                f"b={b.item():.4f}"
            )

    print("\nExpected: w ~= 2, b ~= 1")


if __name__ == "__main__":
    main()
