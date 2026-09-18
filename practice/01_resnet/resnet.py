"""ResNet 논문 실습 코드.

Residual learning의 핵심인 F(x), shortcut x, F(x)+x와
plain network 대비 optimization 차이를 확인하기 위한 공부용 코드다.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionAShortcut(nn.Module):
    """Paper option A: spatial subsampling + zero padding, no learned weights."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        if self.stride == 2:
            x = x[:, :, ::2, ::2]

        if self.out_channels > self.in_channels:
            channel_gap = self.out_channels - self.in_channels
            left = channel_gap // 2
            right = channel_gap - left
            x = F.pad(x, (0, 0, 0, 0, left, right))

        return x


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = OptionAShortcut(in_channels, out_channels, stride)

    def forward(self, x, return_parts=False):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        residual = self.bn2(out)

        out = residual + identity
        out = F.relu(out)

        if return_parts:
            return out, residual, identity
        return out


class ResNet20(nn.Module):
    """CIFAR ResNet-20: 6n+2 with n=3, channels 16/32/64."""

    def __init__(self, residual=True):
        super().__init__()
        Block = ResidualBlock if residual else PlainBlock

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        self.stage1 = nn.Sequential(
            Block(16, 16),
            Block(16, 16),
            Block(16, 16),
        )

        self.stage2 = nn.Sequential(
            Block(16, 32, stride=2),
            Block(32, 32),
            Block(32, 32),
        )

        self.stage3 = nn.Sequential(
            Block(32, 64, stride=2),
            Block(64, 64),
            Block(64, 64),
        )

        self.fc = nn.Linear(64, 10)

    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.bn1(x)
        stem = F.relu(x)

        stage1 = self.stage1(stem)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)

        pooled = F.avg_pool2d(stage3, kernel_size=8)
        pooled = pooled.flatten(1)
        logits = self.fc(pooled)

        if return_features:
            return logits, {
                "stem": stem,
                "stage1": stage1,
                "stage2": stage2,
                "stage3": stage3,
            }
        return logits


def make_resnet20():
    return ResNet20(residual=True)


def make_plain20():
    return ResNet20(residual=False)
