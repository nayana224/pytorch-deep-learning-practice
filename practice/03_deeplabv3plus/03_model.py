"""DeepLabv3+ 논문 실습 코드.

Atrous convolution, ASPP, low-level feature와 decoder가
semantic segmentation에 어떻게 사용되는지 확인하기 위한 공부용 코드다.
"""

import torch

from deeplabv3plus import DeepLabV3Plus


model = DeepLabV3Plus(num_classes=21, output_stride=16)
model.eval()

x = torch.randn(1, 3, 513, 513)

with torch.no_grad():
    logits, features = model(x, return_features=True)

print("input   :", x.shape)
print("low     :", features["low"].shape)
print("high    :", features["high"].shape)
print("ASPP    :", features["aspp"].shape)
print("low->48 :", features["low48"].shape)
print("concat  :", features["concat"].shape)
print("decoded :", features["decoded"].shape)
print("logits  :", logits.shape)
