"""U-Net 논문 실습 코드.

Contracting path, valid convolution, crop-and-copy skip connection,
expanding path와 segmentation 결과를 확인하기 위한 공부용 코드다.
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """3x3 valid convolution + ReLU, twice."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x


def center_crop(encoder_feature, decoder_feature):
    target_h, target_w = decoder_feature.shape[-2:]
    source_h, source_w = encoder_feature.shape[-2:]

    top = (source_h - target_h) // 2
    left = (source_w - target_w) // 2

    return encoder_feature[
        :,
        :,
        top : top + target_h,
        left : left + target_w,
    ]


def center_crop_target(target, logits):
    target_h, target_w = logits.shape[-2:]
    source_h, source_w = target.shape[-2:]

    top = (source_h - target_h) // 2
    left = (source_w - target_w) // 2

    return target[:, top : top + target_h, left : left + target_w]


def print_feature_shapes(features):
    for name, feature in features.items():
        print(f"{name:>10}: {tuple(feature.shape)}")


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()

        # Contracting path
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottom
        self.bottom = DoubleConv(512, 1024)

        # Expanding path
        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = DoubleConv(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = DoubleConv(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = DoubleConv(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x, return_features=False):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottom
        bottom = self.bottom(self.pool(e4))

        # Decoder stage 4
        up4 = self.up4(bottom)
        crop4 = center_crop(e4, up4)
        concat4 = torch.cat([crop4, up4], dim=1)
        d4 = self.dec4(concat4)

        # Decoder stage 3
        up3 = self.up3(d4)
        crop3 = center_crop(e3, up3)
        concat3 = torch.cat([crop3, up3], dim=1)
        d3 = self.dec3(concat3)

        # Decoder stage 2
        up2 = self.up2(d3)
        crop2 = center_crop(e2, up2)
        concat2 = torch.cat([crop2, up2], dim=1)
        d2 = self.dec2(concat2)

        # Decoder stage 1
        up1 = self.up1(d2)
        crop1 = center_crop(e1, up1)
        concat1 = torch.cat([crop1, up1], dim=1)
        d1 = self.dec1(concat1)

        logits = self.final(d1)

        if return_features:
            return logits, {
                "enc1": e1,
                "enc2": e2,
                "enc3": e3,
                "enc4": e4,
                "bottleneck": bottom,
                "up4": up4,
                "crop4": crop4,
                "concat4": concat4,
                "dec4": d4,
                "up3": up3,
                "crop3": crop3,
                "concat3": concat3,
                "dec3": d3,
                "up2": up2,
                "crop2": crop2,
                "concat2": concat2,
                "dec2": d2,
                "up1": up1,
                "crop1": crop1,
                "concat1": concat1,
                "dec1": d1,
                "logits": logits,
            }

        return logits
