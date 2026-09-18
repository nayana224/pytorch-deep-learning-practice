"""DINOv2 실습 파일.

공식 pretrained feature를 사용해 논문의 표현 학습 특성을 관찰한다.
"""

import torch
from torchvision.datasets import OxfordIIITPet

from common import extract_features, image_transform, load_model


model, device = load_model()
transform = image_transform()

try:
    dataset = OxfordIIITPet(
        "data/07_dinov2",
        split="test",
        download=False,
        transform=transform,
    )
except RuntimeError as error:
    raise FileNotFoundError(
        "Oxford-IIIT Pets is not prepared. Run: "
        "python scripts/download_torchvision_data.py pets"
    ) from error

image, label = dataset[0]
image = image.unsqueeze(0).to(device)

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
