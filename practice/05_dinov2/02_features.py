import torch
from torchvision.datasets import OxfordIIITPet

from common import extract_features, image_transform, load_model


# 1. Official pretrained DINOv2
model, device = load_model()

# 2. Paper benchmark image
transform = image_transform()
dataset = OxfordIIITPet(
    "data/05_dinov2",
    split="test",
    download=True,
    transform=transform,
)

image, label = dataset[0]
image = image.unsqueeze(0).to(device)

# 3. image -> CLS feature + patch features
with torch.no_grad():
    class_token, patch_tokens = extract_features(model, image)

num_patches = patch_tokens.shape[1]
patch_grid = int(num_patches ** 0.5)

print("input        :", image.shape)
print("CLS token    :", class_token.shape)
print("patch tokens :", patch_tokens.shape)
print("patch grid   :", patch_grid, "x", patch_grid)
print("CLS norm     :", class_token.norm(dim=-1).item())
print("patch norm   :", patch_tokens.norm(dim=-1).mean().item())
