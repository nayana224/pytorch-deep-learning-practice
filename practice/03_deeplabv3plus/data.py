from pathlib import Path
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation


DATA_DIR = Path("data/03_deeplabv3plus")


def pair_to_tensor(image, mask, train=False, crop_size=513):
    if train:
        scale = random.uniform(0.5, 2.0)

        scaled_h = max(crop_size, int(image.height * scale))
        scaled_w = max(crop_size, int(image.width * scale))

        image = TF.resize(image, [scaled_h, scaled_w])
        mask = TF.resize(
            mask,
            [scaled_h, scaled_w],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        top = torch.randint(0, scaled_h - crop_size + 1, ()).item()
        left = torch.randint(0, scaled_w - crop_size + 1, ()).item()

        image = TF.crop(image, top, left, crop_size, crop_size)
        mask = TF.crop(mask, top, left, crop_size, crop_size)

        if torch.rand(()) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

    image_tensor = TF.to_tensor(image)
    image_tensor = TF.normalize(
        image_tensor,
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    )

    mask_tensor = torch.from_numpy(np.array(mask)).long()
    return image_tensor, mask_tensor


class VOC2012Dataset(Dataset):
    def __init__(self, split="train", train_transform=False):
        self.dataset = VOCSegmentation(
            DATA_DIR,
            year="2012",
            image_set=split,
            download=True,
        )
        self.train_transform = train_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, mask = self.dataset[index]
        return pair_to_tensor(
            image,
            mask,
            train=self.train_transform,
        )
