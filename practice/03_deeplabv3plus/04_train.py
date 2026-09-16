import argparse
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import VOCSegmentation

from deeplabv3plus import DeepLabV3Plus

D = SourceFileLoader(
    "voc_data",
    "practice/03_deeplabv3plus/01_data.py",
).load_module()


class VOCDataset(Dataset):
    def __init__(self, split, train):
        self.ds = VOCSegmentation(
            "data/03_deeplabv3plus",
            year="2012",
            image_set=split,
            download=True,
        )
        self.train = train

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return D.pair_to_tensor(*self.ds[i], train=self.train)


def confusion(pred, gt, n=21):
    mask = gt != 255
    return torch.bincount(
        (gt[mask] * n + pred[mask]).view(-1),
        minlength=n * n,
    ).reshape(n, n)


def miou(hist):
    inter = hist.diag()
    union = hist.sum(1) + hist.sum(0) - inter
    return (inter[union > 0] / union[union > 0].clamp_min(1)).mean().item()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=4)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = DataLoader(
        VOCDataset("train", True),
        batch_size=a.batch_size,
        shuffle=True,
        num_workers=4,
    )
    # VOC validation images keep their original, variable spatial sizes.
    # batch_size=1 avoids padding/resize that would change the evaluation image.
    val = DataLoader(
        VOCDataset("val", False),
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    model = DeepLabV3Plus(21, 16).to(dev)
    opt = torch.optim.SGD(
        model.parameters(),
        lr=0.007,
        momentum=0.9,
        weight_decay=4e-5,
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    out = Path("outputs/03_deeplabv3plus")
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(a.epochs):
        model.train()
        total = 0.0
        for x, y in train:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()

        model.eval()
        hist = torch.zeros(21, 21, device=dev)
        with torch.no_grad():
            for x, y in val:
                x, y = x.to(dev), y.to(dev)
                pred = model(x).argmax(1)
                hist += confusion(pred, y)

        print(
            f"epoch={epoch + 1} "
            f"loss={total / len(train):.4f} "
            f"mIoU={miou(hist):.4f}"
        )

    torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
