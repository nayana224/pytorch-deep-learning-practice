"""DeepLabv3+ 논문 실습 코드.

Atrous convolution, ASPP, low-level feature와 decoder가
semantic segmentation에 어떻게 사용되는지 확인하기 위한 공부용 코드다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AtrousSeparableConv(nn.Module):
    """Depthwise atrous conv followed by 1x1 pointwise conv."""

    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x)


class SimpleXceptionBackbone(nn.Module):
    """Scaled Xception-style backbone for studying the paper data flow."""

    def __init__(self, output_stride=16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.low_level = nn.Sequential(
            AtrousSeparableConv(64, 128),
            nn.MaxPool2d(2, 2),
        )

        stride = 2 if output_stride == 16 else 1
        dilation = 1 if output_stride == 16 else 2

        self.high_level = nn.Sequential(
            AtrousSeparableConv(128, 256),
            nn.Conv2d(256, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                256,
                728,
                3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            AtrousSeparableConv(728, 2048, dilation=dilation),
        )

    def forward(self, x):
        x = self.stem(x)
        low = self.low_level(x)
        high = self.high_level(low)
        return low, high


class ASPP(nn.Module):
    def __init__(self, output_stride=16):
        super().__init__()
        rates = [6, 12, 18] if output_stride == 16 else [12, 24, 36]

        self.branch1 = nn.Conv2d(2048, 256, 1)
        self.branch2 = AtrousSeparableConv(2048, 256, dilation=rates[0])
        self.branch3 = AtrousSeparableConv(2048, 256, dilation=rates[1])
        self.branch4 = AtrousSeparableConv(2048, 256, dilation=rates[2])
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2048, 256, 1),
        )
        self.project = nn.Conv2d(256 * 5, 256, 1)

    def forward(self, x, return_branches=False):
        size = x.shape[-2:]

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        pooled = self.image_pool(x)
        pooled = F.interpolate(
            pooled,
            size=size,
            mode="bilinear",
            align_corners=False,
        )

        concatenated = torch.cat([b1, b2, b3, b4, pooled], dim=1)
        projected = self.project(concatenated)

        if return_branches:
            return projected, {
                "1x1": b1,
                "rate6": b2,
                "rate12": b3,
                "rate18": b4,
                "image_pool": pooled,
            }

        return projected


class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=21, output_stride=16, use_decoder=True):
        super().__init__()

        self.use_decoder = use_decoder
        self.backbone = SimpleXceptionBackbone(output_stride)
        self.aspp = ASPP(output_stride)

        # Paper decoder: low-level feature -> 48 channels
        self.low_reduce = nn.Conv2d(128, 48, 1)

        # 256 ASPP + 48 low-level = 304 channels
        self.decoder_conv1 = AtrousSeparableConv(304, 256)
        self.decoder_conv2 = AtrousSeparableConv(256, 256)
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, x, return_features=False):
        input_size = x.shape[-2:]

        low, high = self.backbone(x)

        if return_features:
            context, aspp_branches = self.aspp(
                high,
                return_branches=True,
            )
        else:
            context = self.aspp(high)
            aspp_branches = None

        if self.use_decoder:
            context_up = F.interpolate(
                context,
                size=low.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            low48 = self.low_reduce(low)
            merged = torch.cat([context_up, low48], dim=1)

            decoded = self.decoder_conv1(merged)
            decoded = self.decoder_conv2(decoded)
            prediction_feature = decoded
        else:
            # DeepLabv3-like baseline for the decoder comparison.
            context_up = context
            low48 = None
            merged = None
            decoded = None
            prediction_feature = context

        logits = self.classifier(prediction_feature)
        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )

        if return_features:
            features = {
                "low": low,
                "high": high,
                "aspp": context,
                "prediction_feature": prediction_feature,
            }

            for name, feature in aspp_branches.items():
                features[f"aspp_{name}"] = feature

            if self.use_decoder:
                features.update(
                    {
                        "aspp_up": context_up,
                        "low48": low48,
                        "concat": merged,
                        "decoded": decoded,
                    }
                )

            return logits, features

        return logits
