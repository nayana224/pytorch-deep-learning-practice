"""
03-1. Residual Block 실습

목표
- notebook 상태에 의존하지 않고 위에서 아래로 한 번에 실행한다.
- 사용자가 직접 타이핑하면서 plain block과 residual block의 차이를 확인한다.
- 완성 코드를 미리 제공하지 않고, 단계별 TODO를 채워간다.

핵심 질문
1. x와 F(x)의 shape은 왜 같아야 하는가?
2. shortcut은 실제 tensor 연산에서 무엇을 하는가?
3. 같은 weight를 쓸 때 residual output과 plain output의 차이는 무엇인가?
4. 마지막에 backward를 했을 때 gradient는 어떻게 계산되는가?

진행 순서
Step 1. 입력 tensor 확인
Step 2. PlainBlock 직접 구현
Step 3. ResidualBlock 직접 구현
Step 4. x, F(x), F(x)+x 확인
Step 5. Plain / Residual의 weight를 동일하게 맞춰 비교
Step 6. projection shortcut으로 shape mismatch 해결
Step 7. 간단한 backward로 gradient 확인

주의
- 한 단계가 확인되기 전 다음 TODO를 한꺼번에 채우지 않는다.
- 비교 실험에서는 seed와 weight를 통제한다.
"""

import torch
import torch.nn as nn


def main():
    # Step 1 -------------------------------------------------------------
    # 먼저 아래 입력 tensor를 실행하고 B, C, H, W가 무엇인지 확인한다.
    torch.manual_seed(0)

    x = torch.randn(4, 16, 32, 32)

    print("x shape:", x.shape)
    print("x dtype:", x.dtype)
    print("x mean:", x.mean().item())

    # Step 2 -------------------------------------------------------------
    # TODO: PlainBlock(nn.Module)을 직접 구현한다.
    # 조건:
    # - Conv2d: 16 -> 16
    # - kernel_size=3
    # - stride=1
    # - padding=1
    # - Conv -> ReLU -> Conv
    #
    # 구현 후 아래를 확인한다.
    # plain_out.shape == x.shape

    # Step 3 -------------------------------------------------------------
    # TODO: ResidualBlock(nn.Module)을 직접 구현한다.
    # residual branch는 Step 2와 같은 Conv -> ReLU -> Conv 구조를 사용한다.
    # 마지막에 identity shortcut x를 더한다.
    #
    # 논문 수식:
    # F(x) = residual branch의 출력
    # y = F(x) + x

    # Step 4 -------------------------------------------------------------
    # TODO: F(x)를 따로 계산해 다음 세 tensor를 비교한다.
    # - x
    # - F(x)
    # - F(x) + x
    #
    # shape뿐 아니라 mean / std도 확인한다.

    # Step 5 -------------------------------------------------------------
    # TODO: PlainBlock과 ResidualBlock의 convolution weight와 bias를 동일하게 맞춘다.
    # 이때 두 block의 유일한 차이가 shortcut이 되도록 만든다.
    #
    # 검증할 것:
    # residual_out - plain_out ≈ x
    #
    # torch.allclose만 보지 말고 max absolute error도 함께 확인한다.

    # Step 6 -------------------------------------------------------------
    # TODO: channel 또는 spatial size가 달라지는 경우를 직접 만든다.
    # 예: 16 -> 32 channels, stride=2
    # identity shortcut으로는 더할 수 없는 이유를 먼저 확인한 뒤,
    # 1x1 convolution projection shortcut을 추가한다.

    # Step 7 -------------------------------------------------------------
    # TODO: 아주 단순한 scalar loss를 만든 뒤 backward()를 호출한다.
    # plain / residual에서 입력 x 또는 첫 convolution weight의 gradient를 확인한다.
    # 여기서는 결과를 일반화하지 말고 "실제로 gradient가 계산되는지"만 먼저 본다.


if __name__ == "__main__":
    main()
