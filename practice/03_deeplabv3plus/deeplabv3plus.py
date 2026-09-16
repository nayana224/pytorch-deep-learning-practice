import torch
import torch.nn as nn
import torch.nn.functional as F


class SepConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, dilation=1):
        super().__init__()
        pad = dilation * (k - 1) // 2
        self.depthwise = nn.Conv2d(in_ch, in_ch, k, stride=stride, padding=pad, dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.relu(self.bn(self.pointwise(self.depthwise(x))), inplace=True)


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, reps=2):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(reps):
            layers.append(SepConv(ch, out_ch, dilation=dilation))
            ch = out_ch
        self.body = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch)) if (in_ch != out_ch or stride != 1) else nn.Identity()
        self.stride = stride

    def forward(self, x):
        y = self.body(x)
        if self.stride != 1:
            y = F.max_pool2d(y, self.stride, self.stride)
        return F.relu(y + self.proj(x), inplace=True)


class ScaledXception65(nn.Module):
    """Paper-structured Xception backbone with fewer middle-flow blocks for local study."""

    def __init__(self, output_stride=16):
        super().__init__()
        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.entry1 = XceptionBlock(64, 128, stride=2, reps=2)  # OS=4, low-level
        self.entry2 = XceptionBlock(128, 256, stride=2, reps=2) # OS=8
        stride3 = 2 if output_stride == 16 else 1
        middle_dilation = 1 if output_stride == 16 else 2
        self.entry3 = XceptionBlock(256, 728, stride=stride3, reps=2)
        self.middle = nn.Sequential(*[XceptionBlock(728, 728, dilation=middle_dilation, reps=3) for _ in range(4)])
        self.exit = nn.Sequential(
            SepConv(728, 1024, dilation=middle_dilation),
            SepConv(1024, 1536, dilation=middle_dilation * 2),
            SepConv(1536, 2048, dilation=middle_dilation * 2),
        )

    def forward(self, x):
        x = self.stem(x)
        low = self.entry1(x)
        x = self.entry2(low)
        x = self.entry3(x)
        x = self.middle(x)
        high = self.exit(x)
        return low, high


class ASPP(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256, output_stride=16):
        super().__init__()
        rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]
        self.b0 = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.bs = nn.ModuleList([SepConv(in_ch, out_ch, dilation=r) for r in rates])
        self.image_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.ReLU(inplace=True))
        self.project = nn.Sequential(nn.Conv2d(out_ch * 5, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True), nn.Dropout(0.1))

    def forward(self, x):
        size = x.shape[-2:]
        parts = [self.b0(x)] + [b(x) for b in self.bs]
        pooled = F.interpolate(self.image_pool(x), size=size, mode="bilinear", align_corners=False)
        return self.project(torch.cat(parts + [pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, output_stride=16):
        super().__init__()
        self.backbone = ScaledXception65(output_stride)
        self.aspp = ASPP(output_stride=output_stride)
        self.low_reduce = nn.Sequential(nn.Conv2d(128, 48, 1, bias=False), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.decoder = nn.Sequential(SepConv(304, 256), SepConv(256, 256))
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, return_features=False):
        input_size = x.shape[-2:]
        low, high = self.backbone(x)
        context = self.aspp(high)
        context_up = F.interpolate(context, size=low.shape[-2:], mode="bilinear", align_corners=False)
        low48 = self.low_reduce(low)
        merged = torch.cat([context_up, low48], dim=1)
        decoded = self.decoder(merged)
        logits = self.classifier(decoded)
        logits = F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        if return_features:
            return logits, {"low": low, "high": high, "aspp": context, "low48": low48, "merged": merged, "decoded": decoded}
        return logits
