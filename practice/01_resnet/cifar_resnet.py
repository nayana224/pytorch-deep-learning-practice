import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionAShortcut(nn.Module):
    """CIFAR-10 option A shortcut from the ResNet paper."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

    def forward(self, x):
        if self.stride == 2:
            x = x[:, :, ::2, ::2]
        if self.out_channels > self.in_channels:
            pad = self.out_channels - self.in_channels
            left = pad // 2
            right = pad - left
            x = F.pad(x, (0, 0, 0, 0, left, right))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = OptionAShortcut(in_channels, out_channels, stride)

    def forward(self, x, return_residual=False):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        residual = self.bn2(self.conv2(out))
        out = F.relu(residual + self.shortcut(x), inplace=True)
        if return_residual:
            return out, residual
        return out


class PlainBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        return F.relu(self.bn2(self.conv2(x)), inplace=True)


class CIFARNet(nn.Module):
    """Paper CIFAR architecture: 6n+2 weighted layers, filters 16/32/64."""

    def __init__(self, depth=20, residual=True, num_classes=10):
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth must satisfy 6n+2, e.g. 20, 32, 44, 56, 110")
        n = (depth - 2) // 6
        block = ResidualBlock if residual else PlainBlock
        self.residual = residual
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )
        self.stage1 = self._make_stage(block, 16, 16, n, stride=1)
        self.stage2 = self._make_stage(block, 16, 32, n, stride=2)
        self.stage3 = self._make_stage(block, 32, 64, n, stride=2)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    @staticmethod
    def _make_stage(block, in_channels, out_channels, n, stride):
        layers = [block(in_channels, out_channels, stride=stride)]
        layers += [block(out_channels, out_channels) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        f0 = self.stem(x)
        f1 = self.stage1(f0)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        pooled = F.adaptive_avg_pool2d(f3, 1).flatten(1)
        logits = self.fc(pooled)
        if return_features:
            return logits, {"stem": f0, "stage1": f1, "stage2": f2, "stage3": f3}
        return logits


def resnet_cifar(depth=20):
    return CIFARNet(depth=depth, residual=True)


def plain_cifar(depth=20):
    return CIFARNet(depth=depth, residual=False)
