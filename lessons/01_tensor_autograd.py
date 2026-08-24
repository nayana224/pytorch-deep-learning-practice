import torch


def main() -> None:
    """Tensor shape과 autograd의 기본 동작을 확인한다."""
    x = torch.tensor([1.0, 2.0, 3.0])
    w = torch.tensor([0.5, -1.0, 2.0], requires_grad=True)
    b = torch.tensor(0.3, requires_grad=True)

    z = torch.dot(w, x) + b
    y_hat = torch.sigmoid(z)

    print(f"x.shape: {x.shape}")
    print(f"w.shape: {w.shape}")
    print(f"z: {z.item():.6f}")
    print(f"y_hat: {y_hat.item():.6f}")

    y_hat.backward()

    print("\n=== Gradient ===")
    print(f"w.grad: {w.grad}")
    print(f"b.grad: {b.grad}")

    # sigmoid(w^T x + b)의 미분 결과와 autograd 결과를 직접 비교한다.
    expected_common = y_hat.detach() * (1.0 - y_hat.detach())
    expected_w_grad = expected_common * x
    expected_b_grad = expected_common

    print("\n=== Manual check ===")
    print(f"expected w.grad: {expected_w_grad}")
    print(f"expected b.grad: {expected_b_grad}")


if __name__ == "__main__":
    main()
