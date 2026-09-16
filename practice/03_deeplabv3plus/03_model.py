import torch
from deeplabv3plus import DeepLabV3Plus

if __name__ == "__main__":
    x = torch.randn(1, 3, 513, 513)
    model = DeepLabV3Plus(num_classes=21, output_stride=16).eval()
    with torch.no_grad():
        logits, feats = model(x, return_features=True)
    print("input  :", x.shape)
    for k, v in feats.items(): print(f"{k:7s}:", v.shape)
    print("logits :", logits.shape)
    print("Key decoder check: low-level -> 48 channels, ASPP upsample -> concat -> two 3x3 separable convs")
