from pathlib import Path
import random

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from torchvision.datasets import VOCSegmentation

DATA = Path("data/03_deeplabv3plus")
OUT = Path("outputs/03_deeplabv3plus")
OUT.mkdir(parents=True, exist_ok=True)


def pair_to_tensor(image, mask, train=False, crop=513):
    if train:
        scale = random.uniform(0.5, 2.0)
        h, w = image.height, image.width
        image = TF.resize(image, [max(crop, int(h * scale)), max(crop, int(w * scale))])
        mask = TF.resize(mask, [max(crop, int(h * scale)), max(crop, int(w * scale))], interpolation=TF.InterpolationMode.NEAREST)
        i, j, hh, ww = torch.randint(0, image.height - crop + 1, (1,)).item(), torch.randint(0, image.width - crop + 1, (1,)).item(), crop, crop
        image, mask = TF.crop(image, i, j, hh, ww), TF.crop(mask, i, j, hh, ww)
        if torch.rand(()) < 0.5:
            image, mask = TF.hflip(image), TF.hflip(mask)
    x = TF.to_tensor(image)
    x = TF.normalize(x, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    y = torch.as_tensor(__import__("numpy").array(mask), dtype=torch.long)
    return x, y


if __name__ == "__main__":
    ds = VOCSegmentation(DATA, year="2012", image_set="train", download=True)
    image, mask = ds[0]
    x, y = pair_to_tensor(image, mask, train=False)
    print("dataset:", len(ds), "image:", x.shape, x.dtype, "mask:", y.shape, y.dtype)
    print("classes:", torch.unique(y))
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image)
    axes[0].set_title("VOC 2012 image")
    axes[1].imshow(mask, cmap="tab20")
    axes[1].set_title("semantic GT (255=ignore)")
    for ax in axes: ax.axis("off")
    plt.tight_layout(); plt.savefig(OUT / "01_voc.png", dpi=150); plt.show()
