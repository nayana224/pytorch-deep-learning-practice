import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import VOC2012Dataset
from deeplabv3plus import DeepLabV3Plus


def confusion_matrix(prediction, target, num_classes=21):
    valid = target != 255

    encoded = target[valid] * num_classes + prediction[valid]
    histogram = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )

    return histogram.reshape(num_classes, num_classes)


def mean_iou(histogram):
    intersection = histogram.diag()
    union = histogram.sum(dim=1) + histogram.sum(dim=0) - intersection

    valid = union > 0
    return (intersection[valid] / union[valid]).mean().item()


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=4)
parser.add_argument(
    "--variant",
    choices=["v3plus", "v3"],
    default="v3plus",
    help="v3plus uses the low-level decoder; v3 is the no-decoder baseline.",
)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = VOC2012Dataset(
    split="train",
    train_transform=True,
)
validation_set = VOC2012Dataset(
    split="val",
    train_transform=False,
)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
)

# Validation VOC images have different spatial sizes.
validation_loader = DataLoader(
    validation_set,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

use_decoder = args.variant == "v3plus"

model = DeepLabV3Plus(
    num_classes=21,
    output_stride=16,
    use_decoder=use_decoder,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.007,
    momentum=0.9,
    weight_decay=4e-5,
)

history = {
    "variant": args.variant,
    "train_loss": [],
    "validation_miou": [],
}

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        logits = model(images)
        loss = criterion(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    average_loss = train_loss / len(train_loader)

    model.eval()
    histogram = torch.zeros(21, 21, device=device)

    with torch.no_grad():
        for images, targets in validation_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            prediction = logits.argmax(dim=1)

            histogram += confusion_matrix(
                prediction,
                targets,
            )

    validation_miou = mean_iou(histogram)

    history["train_loss"].append(average_loss)
    history["validation_miou"].append(validation_miou)

    print(
        f"variant={args.variant} "
        f"epoch={epoch + 1} "
        f"loss={average_loss:.4f} "
        f"mIoU={validation_miou:.4f}"
    )

output_dir = Path("outputs/03_deeplabv3plus")
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint = {
    "model_state_dict": model.state_dict(),
    "variant": args.variant,
    "output_stride": 16,
    "history": history,
}
torch.save(checkpoint, output_dir / f"{args.variant}.pt")

with open(
    output_dir / f"04_history_{args.variant}.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(history, file, indent=2)

epochs = range(1, args.epochs + 1)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

axes[0].plot(epochs, history["train_loss"])
axes[0].set_title(f"{args.variant}: train loss")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("cross entropy")

axes[1].plot(epochs, history["validation_miou"])
axes[1].set_title(f"{args.variant}: validation mIoU")
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("mIoU")

fig.tight_layout()
fig.savefig(
    output_dir / f"04_training_curves_{args.variant}.png",
    dpi=160,
)
plt.close(fig)
