"""ViT에서 patch token, CLS token, attention shape을 확인한다.

첫 바퀴에서는 큰 ViT-B/16을 반드시 실행할 필요가 없다.
기본은 같은 data flow를 유지하는 작은 tiny 모델을 사용하고,
원 논문 규모의 ViT-B/16 shape까지 보고 싶을 때만 --include-base를 사용한다.
"""

import argparse

import torch

from vit import vit_b16, vit_tiny16


parser = argparse.ArgumentParser()
parser.add_argument(
    "--include-base",
    action="store_true",
    help="ViT-B/16도 함께 실행한다. 메모리 사용량이 커질 수 있다.",
)
args = parser.parse_args()

# 첫 바퀴의 핵심은 모델 크기가 아니라
# image → patch embedding → CLS/position → self-attention 흐름이다.
models = [
    ("scaled tiny/16", vit_tiny16(num_classes=100, image_size=224), 224),
]

if args.include_base:
    models.insert(
        0,
        ("ViT-B/16", vit_b16(num_classes=100, image_size=384), 384),
    )

for name, model, image_size in models:
    # 실제 학습 데이터가 없어도 구조와 tensor shape은 확인할 수 있다.
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
    print("input              :", tuple(x.shape))
    print("patch embeddings   :", tuple(features["patches"].shape))
    print("tokens + CLS       :", tuple(features["tokens"].shape))
    print("logits             :", tuple(logits.shape))
    print("last attention map :", tuple(features["attentions"][-1].shape))
    print("parameter count    :", sum(p.numel() for p in model.parameters()))
