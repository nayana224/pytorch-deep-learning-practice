"""
03-1. Residual Block 실습

목표
- 위에서 아래로 한 번에 실행한다.
- 사용자가 직접 타이핑하면서 plain block과 residual block의 차이를 확인한다.
- 완성 코드를 미리 제공하지 않고, 단계별 TODO를 채워간다.

핵심 질문
1. x와 F(x)의 shape은 왜 같아야 하는가?
2. shortcut은 실제 tensor 연산에서 무엇을 하는가?
3. 같은 weight를 쓸 때 residual output과 plain output의 차이는 무엇인가?
4. 마지막에 backward를 했을 때 gradient는 어떻게 계산되는가?
"""

import torch
import torch.nn as nn


def main():
    torch.manual_seed(0)
    x = torch.randn(4, 16, 32, 32)
    print("x shape:", x.shape)
    print("x dtype:", x.dtype)
    print("x mean:", x.mean().item())

    # TODO 1: PlainBlock 구현
    # TODO 2: ResidualBlock 구현
    # TODO 3: x, F(x), F(x)+x 확인
    # TODO 4: weight를 같게 맞춘 뒤 plain/residual 비교
    # TODO 5: projection shortcut 구현
    # TODO 6: backward로 gradient 확인


if __name__ == "__main__":
    main()
