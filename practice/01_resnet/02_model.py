"""ResNet 논문 실습 코드.

Residual learning의 핵심인 F(x), shortcut x, F(x)+x와
plain network 대비 optimization 차이를 확인하기 위한 공부용 코드다.
"""

import torch

from resnet import ResidualBlock, make_plain20, make_resnet20


x = torch.randn(1, 3, 32, 32)

plain20 = make_plain20()
resnet20 = make_resnet20()

print("=== Plain-20 ===")
plain_logits, plain_features = plain20(x, return_features=True)
for name, feature in plain_features.items():
    print(name, feature.shape)
print("logits", plain_logits.shape)

print()
print("=== ResNet-20 ===")
res_logits, res_features = resnet20(x, return_features=True)
for name, feature in res_features.items():
    print(name, feature.shape)
print("logits", res_logits.shape)

print()
print("=== One residual block ===")
block = ResidualBlock(16, 16)
feature = torch.randn(1, 16, 32, 32)
out, residual, identity = block(feature, return_parts=True)

print("x        :", feature.shape)
print("F(x)     :", residual.shape)
print("shortcut :", identity.shape)
print("F(x) + x :", out.shape)
