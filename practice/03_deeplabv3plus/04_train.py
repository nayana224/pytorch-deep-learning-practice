import argparse
from pathlib import Path

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

model = DeepLabV3Plus(
    num_classes=21,
    output_stride=16,
).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.007,
    momentum=0.9,
    weight_decay=4e-5,
)

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

    print(
        f"epoch={epoch + 1} "
        f"loss={train_loss / len(train_loader):.4f} "
        f"mIoU={mean_iou(histogram):.4f}"
    )

output_dir = Path("outputs/03_deeplabv3plus")
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_dir / "model.pt")
