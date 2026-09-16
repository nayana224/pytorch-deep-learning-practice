from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as TF


DATA_DIR = Path("data/01_resnet")
OUT_DIR = Path("outputs/01_resnet")
OUT_DIR.mkdir(parents=True, exist_ok=True)


class PaperCIFAR10(Dataset):
    """CIFAR-10 with the paper's per-pixel mean subtraction and simple augmentation."""

    def __init__(self, train=True, augment=False):
        self.base = CIFAR10(DATA_DIR, train=train, download=True)
        train_base = CIFAR10(DATA_DIR, train=True, download=True)
        self.pixel_mean = torch.from_numpy(train_base.data.astype(np.float32).mean(axis=0) / 255.0).permute(2, 0, 1)
        self.augment = augment

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image, target = self.base[idx]
        x = TF.to_tensor(image)
        if self.augment:
            x = TF.pad(x, 4)
            i, j, h, w = torch.randint(0, 9, (1,)).item(), torch.randint(0, 9, (1,)).item(), 32, 32
            x = TF.crop(x, i, j, h, w)
            if torch.rand(()) < 0.5:
                x = TF.hflip(x)
        x = x - self.pixel_mean
        return x, target


if __name__ == "__main__":
    ds = PaperCIFAR10(train=True, augment=False)
    x, y = ds[0]
    print("dataset size:", len(ds))
    print("image:", x.shape, x.dtype, float(x.min()), float(x.max()))
    print("label:", y)
    print("pixel mean:", ds.pixel_mean.shape, float(ds.pixel_mean.mean()))

    raw, _ = ds.base[0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(raw)
    axes[0].set_title("raw CIFAR-10")
    axes[1].imshow((x + ds.pixel_mean).permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title("tensor before mean subtraction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "01_data.png", dpi=150)
    plt.show()
