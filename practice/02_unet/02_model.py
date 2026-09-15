import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=0,
            ),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def center_crop(encoder_feature, decoder_feature):
    target_h = decoder_feature.shape[-2]
    target_w = decoder_feature.shape[-1]

    h = encoder_feature.shape[-2]
    w = encoder_feature.shape[-1]

    top = (h - target_h) // 2
    left = (w - target_w) // 2

    return encoder_feature[
        :,
        :,
        top : top + target_h,
        left : left + target_w,
    ]


class UNetFirstUp(nn.Module):
    def __init__(self):
        super().__init__()

        # Contracting path
        self.enc1 = DoubleConv(1, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.bottleneck = DoubleConv(512, 1024)

        # Expanding path - first stage
        self.up1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2,
        )

        self.dec1 = DoubleConv(1024, 512)

    def forward(self, x):
        # Contracting path
        x1 = self.enc1(x)
        p1 = self.pool(x1)

        x2 = self.enc2(p1)
        p2 = self.pool(x2)

        x3 = self.enc3(p2)
        p3 = self.pool(x3)

        x4 = self.enc4(p3)
        p4 = self.pool(x4)

        x5 = self.bottleneck(p4)

        # Expanding path - first stage
        up1 = self.up1(x5)

        x4_crop = center_crop(x4, up1)

        merged = torch.cat(
            [x4_crop, up1],
            dim=1,
        )

        d1 = self.dec1(merged)

        return x1, x2, x3, x4, x5, up1, x4_crop, merged, d1


model = UNetFirstUp()

x = torch.randn(1, 1, 572, 572)

x1, x2, x3, x4, x5, up1, x4_crop, merged, d1 = model(x)

print("input      :", x.shape)

print()
print("enc1       :", x1.shape)
print("enc2       :", x2.shape)
print("enc3       :", x3.shape)
print("enc4       :", x4.shape)
print("bottom     :", x5.shape)

print()
print("up1        :", up1.shape)
print("x4 crop    :", x4_crop.shape)
print("concat     :", merged.shape)
print("decoder 1  :", d1.shape)