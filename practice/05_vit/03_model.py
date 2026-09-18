"""Vision Transformer 실습 파일.

논문 구조와 핵심 메커니즘을 읽기 쉽게 따라가기 위한 공부용 코드다.
"""

import torch

from vit import vit_b16, vit_tiny16


models = [
    ("ViT-B/16", vit_b16(num_classes=100, image_size=384), 384),
    ("scaled tiny/16", vit_tiny16(num_classes=100, image_size=224), 224),
]

for name, model, image_size in models:
    x = torch.randn(1, 3, image_size, image_size)
    model.eval()

    with torch.no_grad():
        logits, features = model(
            x,
            return_features=True,
            return_attention=True,
        )

    print()
    print("===", name, "===")
    print("input            :", x.shape)
    print("patch embeddings :", features["patches"].shape)
    print("tokens + CLS     :", features["tokens"].shape)
    print("logits           :", logits.shape)
    print("last attention   :", features["attentions"][-1].shape)
    print("parameters       :", sum(p.numel() for p in model.parameters()))
