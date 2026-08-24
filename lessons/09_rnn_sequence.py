from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn


OUTPUT_DIR = Path("outputs/09_rnn_sequence")


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

    def forward(
        self,
        x: torch.Tensor,
        return_sequence: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.rnn(x)
        last_hidden = hidden[-1]
        logits = self.classifier(last_hidden)

        if return_sequence:
            return logits, output
        return logits


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
    torch.manual_seed(0)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_dataset()
    x = x.to(device)
    y = y.to(device)

    model = SequenceClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    print(f"device: {device}")
    print(f"input shape [batch, sequence, feature]: {tuple(x.shape)}")

    fixed_indices = torch.tensor([0, 1], device=device)
    fixed_x = x[fixed_indices]
    fixed_y = y[fixed_indices]

    loss_history: list[float] = []
    accuracy_history: list[float] = []
    snapshot_epochs = {0, 20, 50, 99}

    visualize_hidden_states(model, fixed_x, fixed_y, epoch=0)

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        accuracy = (logits.argmax(dim=1) == y).float().mean().item()
        loss_history.append(loss.item())
        accuracy_history.append(accuracy)

        if epoch % 20 == 0 or epoch == 99:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.item():.4f} "
                f"accuracy={accuracy:.4f}"
            )

        if epoch in snapshot_epochs:
            visualize_hidden_states(model, fixed_x, fixed_y, epoch=epoch + 1)

    plot_training_curves(loss_history, accuracy_history)
    plot_input_sequences(fixed_x, fixed_y)
    print(f"시각화 저장 위치: {OUTPUT_DIR}")
    plt.show()


def visualize_hidden_states(
    model: SequenceClassifier,
    x: torch.Tensor,
    y: torch.Tensor,
    epoch: int,
) -> None:
    """각 time step에서 hidden state가 어떻게 변하는지 heatmap으로 저장한다."""
    was_training = model.training
    model.eval()

    with torch.no_grad():
        _, hidden_sequence = model(x, return_sequence=True)

    if was_training:
        model.train()

    hidden_sequence = hidden_sequence.cpu()
    y = y.cpu()

    figure, axes = plt.subplots(2, 1, figsize=(9, 6))
    for sample_index, axis in enumerate(axes):
        image = axis.imshow(
            hidden_sequence[sample_index].T.numpy(),
            aspect="auto",
            cmap="coolwarm",
        )
        axis.set_xlabel("Time step")
        axis.set_ylabel("Hidden unit")
        axis.set_title(
            f"sample={sample_index}, label={y[sample_index].item()}, epoch={epoch}"
        )
        figure.colorbar(image, ax=axis)

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / f"hidden_states_epoch_{epoch:03d}.png", dpi=150)
    plt.close(figure)


def plot_input_sequences(x: torch.Tensor, y: torch.Tensor) -> None:
    """RNN에 들어가는 증가/감소 sequence의 순서 차이를 그린다."""
    x = x.cpu().squeeze(-1)
    y = y.cpu()

    plt.figure(figsize=(8, 5))
    for index in range(x.shape[0]):
        plt.plot(
            range(x.shape[1]),
            x[index].numpy(),
            marker="o",
            label=f"sample {index}, label={y[index].item()}",
        )

    plt.xlabel("Time step")
    plt.ylabel("Value")
    plt.title("Sequence order determines the class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "input_sequences.png", dpi=150)


def plot_training_curves(
    loss_history: list[float],
    accuracy_history: list[float],
) -> None:
    """RNN의 loss와 accuracy 변화를 그린다."""
    figure, axes = plt.subplots(2, 1, figsize=(8, 7))

    axes[0].plot(loss_history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("RNN training loss")

    axes[1].plot(accuracy_history)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("RNN training accuracy")

    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)


if __name__ == "__main__":
    main()
