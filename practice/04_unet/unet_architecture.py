"""
04-1. U-Net architecture typing practice

이 파일은 U-Net 완성 코드나 TODO 정답지를 미리 제공하지 않는다.
ChatGPT가 대화에서 작은 실행 단위의 실제 코드를 제시하면,
사용자가 직접 이 파일에 타이핑하고 실행 결과를 확인하면서 코드를 누적한다.

진행 원칙
1. 한 번에 작은 코드만 추가한다.
2. 직접 타이핑한다.
3. 실행 전에 tensor shape을 가능하면 예상한다.
4. 실행 결과를 확인하고 이해되지 않는 줄을 질문한다.
5. 이해가 끝난 뒤 다음 코드로 넘어간다.

최종적으로 이 파일에는 사용자가 직접 타이핑한 original U-Net 구조가 남는다.
기준은 U-Net 논문 Figure 1의 valid convolution / max pooling / up-convolution /
copy-and-crop / concatenation / 1x1 convolution 흐름이다.
"""

import torch
import torch.nn as nn


def main():
    torch.manual_seed(0)

    # Step 1 시작점
    # 논문 Figure 1과 같은 572 x 572 grayscale 입력으로 시작한다.
    x = torch.randn(1, 1, 572, 572)

    print("x shape:", x.shape)
    print("x dtype:", x.dtype)


if __name__ == "__main__":
    main()
