from __future__ import annotations

from collections import OrderedDict

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two valid 3x3 convolutions followed by ReLU, as in the U-Net paper."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


def center_crop(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Center-crop source so its H,W match reference.

    The original U-Net needs this because valid convolutions shrink encoder feature maps.
    """

    target_h, target_w = reference.shape[-2:]
    source_h, source_w = source.shape[-2:]

    if source_h < target_h or source_w < target_w:
        raise ValueError(
            f"source feature {tuple(source.shape)} is smaller than reference {tuple(reference.shape)}"
        )

    top = (source_h - target_h) // 2
    left = (source_w - target_w) // 2

    return source[:, :, top : top + target_h, left : left + target_w]


def center_crop_target(target: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Center-crop segmentation target [B,H,W] to model output spatial size."""

    target_h, target_w = output.shape[-2:]
    source_h, source_w = target.shape[-2:]

    if source_h < target_h or source_w < target_w:
        raise ValueError(
            f"target {tuple(target.shape)} is smaller than model output {tuple(output.shape)}"
        )

    top = (source_h - target_h) // 2
    left = (source_w - target_w) // 2

    return target[:, top : top + target_h, left : left + target_w]


class UNet(nn.Module):
    """Original-style U-Net with valid convolutions and crop+concat skip connections."""

    def __init__(self, in_channels: int = 1, num_classes: int = 2):
        super().__init__()

        # Contracting path
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        # Expanding path
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ):
        features = OrderedDict()
        features["input"] = x

        # Encoder
        e1 = self.enc1(x)
        features["enc1"] = e1
        p1 = self.pool(e1)
        features["pool1"] = p1

        e2 = self.enc2(p1)
        features["enc2"] = e2
        p2 = self.pool(e2)
        features["pool2"] = p2

        e3 = self.enc3(p2)
        features["enc3"] = e3
        p3 = self.pool(e3)
        features["pool3"] = p3

        e4 = self.enc4(p3)
        features["enc4"] = e4
        p4 = self.pool(e4)
        features["pool4"] = p4

        b = self.bottleneck(p4)
        features["bottleneck"] = b

        # Decoder stage 4
        u4 = self.up4(b)
        features["up4"] = u4
        c4 = center_crop(e4, u4)
        features["crop4"] = c4
        m4 = torch.cat([c4, u4], dim=1)
        features["concat4"] = m4
        d4 = self.dec4(m4)
        features["dec4"] = d4

        # Decoder stage 3
        u3 = self.up3(d4)
        features["up3"] = u3
        c3 = center_crop(e3, u3)
        features["crop3"] = c3
        m3 = torch.cat([c3, u3], dim=1)
        features["concat3"] = m3
        d3 = self.dec3(m3)
        features["dec3"] = d3

        # Decoder stage 2
        u2 = self.up2(d3)
        features["up2"] = u2
        c2 = center_crop(e2, u2)
        features["crop2"] = c2
        m2 = torch.cat([c2, u2], dim=1)
        features["concat2"] = m2
        d2 = self.dec2(m2)
        features["dec2"] = d2

        # Decoder stage 1
        u1 = self.up1(d2)
        features["up1"] = u1
        c1 = center_crop(e1, u1)
        features["crop1"] = c1
        m1 = torch.cat([c1, u1], dim=1)
        features["concat1"] = m1
        d1 = self.dec1(m1)
        features["dec1"] = d1

        logits = self.classifier(d1)
        features["logits"] = logits

        if return_features:
            return logits, features
        return logits


def print_feature_shapes(features: OrderedDict[str, torch.Tensor]) -> None:
    for name, tensor in features.items():
        print(f"{name:>10}: {tuple(tensor.shape)}")
