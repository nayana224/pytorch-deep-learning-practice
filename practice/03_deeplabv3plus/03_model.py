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
