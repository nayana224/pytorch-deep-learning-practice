"""U-Net 논문 실습 코드.

Contracting path, valid convolution, crop-and-copy skip connection,
expanding path와 segmentation 결과를 확인하기 위한 공부용 코드다.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import tifffile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from unet import UNet, center_crop_target


DATA_DIR = Path("data/02_unet/isbi2012")
IMAGE_PATH = DATA_DIR / "train-volume.tif"
LABEL_PATH = DATA_DIR / "train-labels.tif"
OUTPUT_DIR = Path("outputs/02_unet")
CHECKPOINT_PATH = OUTPUT_DIR / "unet_baseline.pt"


class ISBI2012Dataset(Dataset):
    def __init__(self, augment: bool = False):
        self.images = tifffile.imread(IMAGE_PATH)
        self.labels = tifffile.imread(LABEL_PATH)
        self.augment = augment

        if self.images.shape != self.labels.shape:
            raise ValueError(
                f"image/label shape mismatch: {self.images.shape} vs {self.labels.shape}"
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image = torch.from_numpy(self.images[index].copy()).float() / 255.0
        image = image.unsqueeze(0)  # [1,H,W]

        label = torch.from_numpy(self.labels[index].copy())
        label = (label == 255).long()  # 0=membrane, 1=cell interior

        if self.augment:
            # Lightweight geometric augmentation for the baseline.
            # The paper's elastic deformation is intentionally left as a separate fidelity step.
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                label = torch.flip(label, dims=[1])

            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                label = torch.flip(label, dims=[0])

            k = random.randint(0, 3)
            image = torch.rot90(image, k, dims=[1, 2])
            label = torch.rot90(label, k, dims=[0, 1])

        return image.contiguous(), label.contiguous()


def membrane_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred_membrane = pred == 0
    target_membrane = target == 0

    intersection = (pred_membrane & target_membrane).sum().item()
    union = (pred_membrane | target_membrane).sum().item()

    return intersection / union if union > 0 else 1.0


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_iou = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        aligned_targets = center_crop_target(targets, logits)
        loss = criterion(logits, aligned_targets)

        if training:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            prediction = logits.argmax(dim=1)
            iou = membrane_iou(prediction, aligned_targets)

        total_loss += loss.item()
        total_iou += iou

    n = len(loader)
    return total_loss / n, total_iou / n


def parse_args():
    parser = argparse.ArgumentParser(description="Train an original-style U-Net baseline.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Keep the split simple and explicit for study purposes.
    # The volume is serially sectioned, so this is not a claim of statistical independence.
    train_indices = list(range(24))
    val_indices = list(range(24, 30))

    train_dataset = Subset(ISBI2012Dataset(augment=True), train_indices)
    val_dataset = Subset(ISBI2012Dataset(augment=False), val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNet(in_channels=1, num_classes=2).to(device)

    # Paper: pixel-wise softmax + cross entropy and SGD with high momentum.
    # This baseline keeps CrossEntropyLoss but does NOT yet reproduce the paper's
    # boundary-aware per-pixel weight map.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.99,
    )

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_iou": [],
        "val_iou": [],
    }

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_iou = run_epoch(
            model, train_loader, criterion, device, optimizer
        )

        with torch.no_grad():
            val_loss, val_iou = run_epoch(
                model, val_loader, criterion, device, optimizer=None
            )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"train membrane IoU {train_iou:.4f} | val membrane IoU {val_iou:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_membrane_iou": val_iou,
                },
                CHECKPOINT_PATH,
            )
            print("  saved:", CHECKPOINT_PATH)

    epochs = range(1, args.epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].set_title("Cross-entropy loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, history["train_iou"], label="train")
    axes[1].plot(epochs, history["val_iou"], label="val")
    axes[1].set_title("Membrane IoU")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_train_curves.png", dpi=160)
    plt.show()


if __name__ == "__main__":
    main()
