import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet

from common import extract_features, image_transform, load_model


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

model, device = load_model()
transform = image_transform()

train_set = OxfordIIITPet(
    "data/05_dinov2",
    split="trainval",
    download=True,
    transform=transform,
)

test_set = OxfordIIITPet(
    "data/05_dinov2",
    split="test",
    download=True,
    transform=transform,
)

train_loader = DataLoader(
    train_set,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
)

test_loader = DataLoader(
    test_set,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
)

# Freeze DINOv2 completely.
for parameter in model.parameters():
    parameter.requires_grad = False

sample_images, _ = next(iter(train_loader))
sample_images = sample_images[:1].to(device)

with torch.no_grad():
    sample_feature, _ = extract_features(model, sample_images)

feature_dim = sample_feature.shape[-1]
probe = nn.Linear(feature_dim, 37).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    probe.parameters(),
    lr=0.05,
    momentum=0.9,
)

for epoch in range(args.epochs):
    probe.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            class_features, _ = extract_features(model, images)

        logits = probe(class_features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    probe.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            class_features, _ = extract_features(model, images)
            logits = probe(class_features)
            prediction = logits.argmax(dim=1)

            correct += (prediction == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total
    print(
        f"epoch={epoch + 1} "
        f"loss={train_loss / len(train_loader):.4f} "
        f"test_acc={accuracy:.4f}"
    )

output_dir = Path("outputs/05_dinov2")
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(probe.state_dict(), output_dir / "pets_linear_probe.pt")
