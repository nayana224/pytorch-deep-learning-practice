import torch
from torch import nn


class SequenceClassifier(nn.Module):
    """숫자 sequence의 순서 정보를 이용하는 간단한 RNN 분류기."""

    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.rnn(x)
        last_hidden = hidden[-1]
        return self.classifier(last_hidden)


def make_dataset(samples: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
    """증가/감소하는 sequence를 만들어 순서가 label을 결정하게 한다."""
    torch.manual_seed(0)

    sequences = []
    labels = []

    for index in range(samples):
        start = torch.rand(1).item()
        step = 0.1 + 0.1 * torch.rand(1).item()

        if index % 2 == 0:
            sequence = torch.tensor(
                [start + step * i for i in range(6)],
                dtype=torch.float32,
            )
            label = 1
        else:
            sequence = torch.tensor(
                [start + step * (5 - i) for i in range(6)],
                dtype=torch.float32,
            )
            label = 0

        sequences.append(sequence.unsqueeze(1))
        labels.append(label)

    return torch.stack(sequences), torch.tensor(labels)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_dataset()
    x = x.to(device)
    y = y.to(device)

    model = SequenceClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print(f"input shape [batch, sequence, feature]: {tuple(x.shape)}")

    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 99:
            accuracy = (logits.argmax(dim=1) == y).float().mean().item()
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"accuracy={accuracy:.4f}"
            )


if __name__ == "__main__":
    main()
