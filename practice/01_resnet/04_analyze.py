from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data import PaperCIFAR10
from resnet import make_plain20, make_resnet20


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = Path("outputs/01_resnet")

models = {
    "plain20": make_plain20(),
    "resnet20": make_resnet20(),
}

test_set = PaperCIFAR10(train=False, augment=False)
test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

for name, model in models.items():
    checkpoint = output_dir / f"{name}.pt"
    if not checkpoint.exists():
        print(f"skip {name}: {checkpoint} not found")
        continue

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
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

    error = 100.0 * (1.0 - correct / total)
    print(f"{name} test error: {error:.2f}%")

images, labels = next(iter(test_loader))
model = models["resnet20"]
checkpoint = output_dir / "resnet20.pt"

if checkpoint.exists():
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(images.to(device))
        prediction = logits.argmax(dim=1).cpu()

    fig, axes = plt.subplots(2, 4, figsize=(10, 5))

    for index, ax in enumerate(axes.flat):
        image = images[index] + test_set.mean_image
        image = image.permute(1, 2, 0).clamp(0, 1)
        ax.imshow(image)
        ax.set_title(f"GT {labels[index]} / Pred {prediction[index]}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()
