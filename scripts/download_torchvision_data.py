from pathlib import Path
import argparse

from torchvision.datasets import CIFAR10, CIFAR100, OxfordIIITPet


def ensure_cifar10():
    root = Path("data/01_resnet")
    try:
        CIFAR10(root, train=True, download=False)
        CIFAR10(root, train=False, download=False)
        print("[skip] CIFAR-10 already exists:", root)
        return
    except RuntimeError:
        pass

    print("[download] CIFAR-10")
    CIFAR10(root, train=True, download=True)
    CIFAR10(root, train=False, download=True)


def ensure_cifar100():
    root = Path("data/04_vit")
    try:
        CIFAR100(root, train=True, download=False)
        CIFAR100(root, train=False, download=False)
        print("[skip] CIFAR-100 already exists:", root)
        return
    except RuntimeError:
        pass

    print("[download] CIFAR-100")
    CIFAR100(root, train=True, download=True)
    CIFAR100(root, train=False, download=True)


def ensure_pets():
    root = Path("data/05_dinov2")
    try:
        OxfordIIITPet(root, split="trainval", download=False)
        OxfordIIITPet(root, split="test", download=False)
        print("[skip] Oxford-IIIT Pets already exists:", root)
        return
    except RuntimeError:
        pass

    print("[download] Oxford-IIIT Pets")
    OxfordIIITPet(root, split="trainval", download=True)
    OxfordIIITPet(root, split="test", download=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    choices=["cifar10", "cifar100", "pets", "all"],
)
args = parser.parse_args()

if args.dataset in ("cifar10", "all"):
    ensure_cifar10()
if args.dataset in ("cifar100", "all"):
    ensure_cifar100()
if args.dataset in ("pets", "all"):
    ensure_pets()
