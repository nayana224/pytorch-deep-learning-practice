"""
ResNet residual block 실습 이력.

이 파일은 사용자가 이미 직접 타이핑한 내용을 보존한다.
향후 코드는 TODO scaffold가 아니라 대화에서 받은 실제 코드를 직접 타이핑하며 이어간다.
"""

import torch
import torch.nn as nn


def main():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 32, 32)
    print("x shape:", x.shape)
    print("x dtype:", x.dtype)
    print("x mean:", x.mean().item())

    class PlainBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            return out

    plain_block = PlainBlock()
    plain_out = plain_block(x)
    print("plain_out shape:", plain_out.shape)

    class ResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            out = self.conv1(x)
            out = self.relu(out)
            out = self.conv2(out)
            out = out + x
            return out

    res_block = ResidualBlock()
    res_out = res_block(x)
    print("res_out shape:", res_out.shape)


if __name__ == "__main__":
    main()
