"""논문 실습에서 사용하는 torchvision dataset을 내려받는다.

공부 코드 자체에서는 download=False를 유지하고,
실제 다운로드는 이 스크립트에서만 수행한다.
"""

from pathlib import Path
import argparse

from torchvision.datasets import CIFAR10, CIFAR100, OxfordIIITPet


def ensure_cifar10_resnet():
    """ResNet 실습용 CIFAR-10을 준비한다."""
    root = Path("data/01_resnet")

    try:
        CIFAR10(root, train=True, download=False)
        CIFAR10(root, train=False, download=False)
        print("[skip] CIFAR-10 for ResNet:", root)
        return
    except RuntimeError:
        pass

    print("[download] CIFAR-10 for ResNet")
    CIFAR10(root, train=True, download=True)
    CIFAR10(root, train=False, download=True)


def ensure_cifar100_vit():
    """ViT 실습용 CIFAR-100을 준비한다."""
    root = Path("data/05_vit")

    try:
        CIFAR100(root, train=True, download=False)
        CIFAR100(root, train=False, download=False)
        print("[skip] CIFAR-100 for ViT:", root)
        return
    except RuntimeError:
        pass

    print("[download] CIFAR-100 for ViT")
    CIFAR100(root, train=True, download=True)
    CIFAR100(root, train=False, download=True)


def ensure_pets_dinov2():
    """DINOv2 feature 분석용 Oxford-IIIT Pets를 준비한다."""
    root = Path("data/07_dinov2")

    try:
        OxfordIIITPet(root, split="trainval", download=False)
        OxfordIIITPet(root, split="test", download=False)
        print("[skip] Oxford-IIIT Pets for DINOv2:", root)
        return
    except RuntimeError:
        pass

    print("[download] Oxford-IIIT Pets for DINOv2")
    OxfordIIITPet(root, split="trainval", download=True)
    OxfordIIITPet(root, split="test", download=True)


def ensure_cifar10_ddpm():
    """DDPM forward/noise 실습용 CIFAR-10을 준비한다."""
    root = Path("data/09_ddpm")

    try:
        CIFAR10(root, train=True, download=False)
        print("[skip] CIFAR-10 for DDPM:", root)
        return
    except RuntimeError:
        pass

    print("[download] CIFAR-10 for DDPM")
    CIFAR10(root, train=True, download=True)


parser = argparse.ArgumentParser()
parser.add_argument(
    "dataset",
    choices=["cifar10", "cifar100", "pets", "ddpm", "all"],
)
args = parser.parse_args()

if args.dataset in ("cifar10", "all"):
    ensure_cifar10_resnet()

if args.dataset in ("cifar100", "all"):
    ensure_cifar100_vit()

if args.dataset in ("pets", "all"):
    ensure_pets_dinov2()

if args.dataset in ("ddpm", "all"):
    ensure_cifar10_ddpm()
