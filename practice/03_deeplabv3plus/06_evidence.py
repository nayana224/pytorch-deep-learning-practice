"""DeepLabv3+ 논문 실습 코드.

Atrous convolution, ASPP, low-level feature와 decoder가
semantic segmentation에 어떻게 사용되는지 확인하기 위한 공부용 코드다.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import VOC2012Dataset
from deeplabv3plus import DeepLabV3Plus


OUTPUT_DIR = Path("outputs/03_deeplabv3plus")


def confusion_matrix(prediction, target, num_classes=21):
    valid = target != 255
    encoded = target[valid] * num_classes + prediction[valid]

    histogram = torch.bincount(
        encoded,
        minlength=num_classes * num_classes,
    )
    return histogram.reshape(num_classes, num_classes)


def mean_iou(histogram):
    intersection = histogram.diag()
    union = histogram.sum(dim=1) + histogram.sum(dim=0) - intersection
    valid = union > 0
    return (intersection[valid] / union[valid]).mean().item()


def semantic_boundary(target):
    boundary = torch.zeros_like(target, dtype=torch.bool)
    valid = target != 255

    boundary[:, 1:] |= target[:, 1:] != target[:, :-1]
    boundary[:, :-1] |= target[:, :-1] != target[:, 1:]
    boundary[:, :, 1:] |= target[:, :, 1:] != target[:, :, :-1]
    boundary[:, :, :-1] |= target[:, :, :-1] != target[:, :, 1:]

    return boundary & valid


def load_model(variant, device):
    checkpoint_path = OUTPUT_DIR / f"{variant}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"{checkpoint_path} not found. Run 04_train.py "
            f"--variant {variant} first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = DeepLabV3Plus(
        num_classes=21,
        output_stride=16,
        use_decoder=variant == "v3plus",
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate(model, loader, device, max_samples):
    histogram = torch.zeros(21, 21, device=device)

    boundary_correct = 0
    boundary_total = 0
    interior_correct = 0
    interior_total = 0

    with torch.no_grad():
        for index, (images, targets) in enumerate(loader):
            if index >= max_samples:
                break

            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            prediction = logits.argmax(dim=1)

            histogram += confusion_matrix(prediction, targets)

            valid = targets != 255
            boundary = semantic_boundary(targets)
            interior = valid & ~boundary

            boundary_correct += (
                (prediction == targets) & boundary
            ).sum().item()
            boundary_total += boundary.sum().item()

            interior_correct += (
                (prediction == targets) & interior
            ).sum().item()
            interior_total += interior.sum().item()

    return {
        "mIoU": mean_iou(histogram),
        "boundary_accuracy": boundary_correct / max(boundary_total, 1),
        "interior_accuracy": interior_correct / max(interior_total, 1),
        "samples": min(max_samples, len(loader)),
    }


parser = argparse.ArgumentParser()
parser.add_argument(
    "--max-samples",
    type=int,
    default=100,
    help="Use a validation subset for a fast paper-claim check.",
)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = VOC2012Dataset(
    split="val",
    train_transform=False,
)
loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
)

models = {
    "DeepLabv3 (no decoder)": load_model("v3", device),
    "DeepLabv3+ (decoder)": load_model("v3plus", device),
}

results = {}

for name, model in models.items():
    results[name] = evaluate(
        model,
        loader,
        device,
        args.max_samples,
    )

    print(name)
    for metric, value in results[name].items():
        print(f"  {metric}: {value}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(
    OUTPUT_DIR / "06_decoder_evidence.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(results, file, indent=2)

metric_names = ["mIoU", "boundary_accuracy", "interior_accuracy"]
labels = list(results.keys())

fig, axes = plt.subplots(1, 3, figsize=(13, 4))

for ax, metric in zip(axes, metric_names):
    values = [results[label][metric] for label in labels]
    ax.bar(labels, values)
    ax.set_ylim(0, 1)
    ax.set_title(metric)
    ax.tick_params(axis="x", rotation=15)

fig.suptitle(
    "Decoder evidence: same scaled backbone/ASPP, "
    "with vs without low-level decoder"
)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "06_decoder_evidence.png", dpi=160)
plt.show()
