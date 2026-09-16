from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as TF


DATA_DIR = Path("data/01_resnet")


class PaperCIFAR10(Dataset):
    """CIFAR-10 preprocessing used in the paper's CIFAR experiment."""

    def __init__(self, train=True, augment=False):
        try:
            self.dataset = CIFAR10(DATA_DIR, train=train, download=False)
            train_set = CIFAR10(DATA_DIR, train=True, download=False)
        except RuntimeError as error:
            raise FileNotFoundError(
                "CIFAR-10 is not prepared. Run: "
                "python scripts/download_torchvision_data.py cifar10"
            ) from error

        self.augment = augment

        mean_image = train_set.data.astype(np.float32).mean(axis=0) / 255.0
        self.mean_image = torch.from_numpy(mean_image).permute(2, 0, 1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = TF.to_tensor(image)

        if self.augment:
            image = TF.pad(image, 4)
            top = torch.randint(0, 9, ()).item()
            left = torch.randint(0, 9, ()).item()
            image = TF.crop(image, top, left, 32, 32)

            if torch.rand(()) < 0.5:
                image = TF.hflip(image)

        image = image - self.mean_image
        return image, label
