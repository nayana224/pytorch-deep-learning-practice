from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.datasets import VOCSegmentation

from data import DATA_DIR, pair_to_tensor
from deeplabv3plus import DeepLabV3Plus


OUTPUT_DIR = Path("outputs/03_deeplabv3plus")
CHECKPOINT = OUTPUT_DIR / "model.pt"

if not CHECKPOINT.exists():
    raise SystemExit("Run practice/03_deeplabv3plus/04_train.py first.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DeepLabV3Plus(num_classes=21, output_stride=16).to(device)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()

try:
    raw_dataset = VOCSegmentation(
        DATA_DIR,
        year="2012",
        image_set="val",
        download=False,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "PASCAL VOC 2012 is not prepared. Run: bash scripts/download_voc2012.sh"
    ) from error

image, mask = raw_dataset[0]
image_tensor, target = pair_to_tensor(image, mask, train=False)

with torch.no_grad():
    logits, features = model(
        image_tensor.unsqueeze(0).to(device),
        return_features=True,
    )

prediction = logits.argmax(dim=1)[0].cpu()
error = (prediction != target) & (target != 255)

print("low-level feature :", features["low"].shape)
print("ASPP feature      :", features["aspp"].shape)
print("low-level -> 48   :", features["low48"].shape)
print("concat            :", features["concat"].shape)
print("decoded           :", features["decoded"].shape)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(image)
axes[0].set_title("input")
axes[1].imshow(target, cmap="tab20")
axes[1].set_title("GT")
axes[2].imshow(prediction, cmap="tab20")
axes[2].set_title("prediction")
axes[3].imshow(error, cmap="gray")
axes[3].set_title("error map")
for ax in axes:
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "05_prediction.png", dpi=150)
plt.show()
