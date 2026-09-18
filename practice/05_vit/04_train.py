"""Vision Transformer 실습 파일.

논문 구조와 핵심 메커니즘을 읽기 쉽게 따라가기 위한 공부용 코드다.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR100

from vit import vit_b16, vit_tiny16


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["base", "tiny"], default="tiny")
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--batch-size", type=int, default=64)
args = parser.parse_args()

image_size = 384 if args.model == "base" else 224

train_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

try:
    train_set = CIFAR100(
        "data/05_vit",
        train=True,
        download=False,
        transform=train_transform,
    )
    test_set = CIFAR100(
        "data/05_vit",
        train=False,
        download=False,
        transform=test_transform,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "CIFAR-100 is not prepared. Run: "
        "python scripts/download_torchvision_data.py cifar100"
    ) from error

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "base":
    model = vit_b16(num_classes=100, image_size=image_size)
else:
    model = vit_tiny16(num_classes=100, image_size=image_size)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
)

for epoch in range(args.epochs):
    model.train()
    train_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            prediction = logits.argmax(dim=1)

            correct += (prediction == labels).sum().item()
            total += labels.numel()

    accuracy = correct / total
    print(
        f"epoch={epoch + 1} "
        f"loss={train_loss / len(train_loader):.4f} "
        f"acc={accuracy:.4f}"
    )

output_dir = Path("outputs/05_vit")
output_dir.mkdir(parents=True, exist_ok=True)

torch.save(
    {
        "model": model.state_dict(),
        "variant": args.model,
        "size": image_size,
    },
    output_dir / f"{args.model}.pt",
)
