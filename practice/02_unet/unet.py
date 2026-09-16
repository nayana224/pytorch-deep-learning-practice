import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """3x3 valid conv + ReLU, twice."""

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
        b = self.bottom(self.pool(e4))

        # Decoder 4: upsample -> crop encoder feature -> concat -> conv
        u4 = self.up4(b)
        c4 = center_crop(e4, u4)
        d4 = self.dec4(torch.cat([c4, u4], dim=1))

        # Decoder 3
        u3 = self.up3(d4)
        c3 = center_crop(e3, u3)
        d3 = self.dec3(torch.cat([c3, u3], dim=1))

        # Decoder 2
        u2 = self.up2(d3)
        c2 = center_crop(e2, u2)
        d2 = self.dec2(torch.cat([c2, u2], dim=1))

        # Decoder 1
        u1 = self.up1(d2)
        c1 = center_crop(e1, u1)
        d1 = self.dec1(torch.cat([c1, u1], dim=1))

        logits = self.final(d1)

        if return_features:
            return logits, {
                "enc1": e1,
                "enc2": e2,
                "enc3": e3,
                "enc4": e4,
                "bottom": b,
                "up4": u4,
                "crop4": c4,
                "dec4": d4,
                "dec3": d3,
                "dec2": d2,
                "dec1": d1,
            }

        return logits
