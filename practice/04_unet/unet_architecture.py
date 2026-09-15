"""
04-1. U-Net architecture 실습

목표
- U-Net 논문의 Figure 1 구조를 작은 PyTorch 코드로 직접 타이핑한다.
- 완성 구현을 복사하지 않고, tensor shape을 먼저 예상한 뒤 한 단계씩 구현한다.
- 이번 파일에서는 학습보다 네트워크 내부 데이터 흐름을 이해하는 데 집중한다.

핵심 질문
1. valid 3x3 convolution을 두 번 통과하면 H, W가 왜 4씩 줄어드는가?
2. encoder에서 spatial size는 줄고 channel 수는 늘어나는 이유는 무엇인가?
3. decoder에서 encoder feature를 왜 다시 가져오는가?
4. ResNet의 element-wise add와 U-Net의 channel-wise concat은 어떻게 다른가?
5. skip feature와 upsampled feature의 spatial size가 다르면 왜 crop이 필요한가?

진행 규칙
- TODO를 한 번에 전부 구현하지 않는다.
- 각 단계에서 실행하고 shape을 기록한 뒤 다음 단계로 넘어간다.
- 예상 shape을 주석으로 먼저 적고 실제 결과와 비교한다.
"""

import torch
import torch.nn as nn


def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------
    # 0. 입력 tensor 확인
    # ------------------------------------------------------------
    # 논문 Figure 1의 입력 크기와 같은 572x572를 사용한다.
    # 처음에는 grayscale biomedical image를 가정하여 channel=1로 둔다.
    x = torch.randn(1, 1, 572, 572)

    print("x shape:", x.shape)
    print("x dtype:", x.dtype)
    print("x min:", x.min().item())
    print("x max:", x.max().item())

    # 실행 전 예상:
    # x shape = ?

    # ------------------------------------------------------------
    # 1. DoubleConv 직접 구현
    # ------------------------------------------------------------
    # 목표 흐름:
    # [B, 1, 572, 572]
    #   -> 3x3 valid conv, 64 channels
    #   -> ReLU
    #   -> 3x3 valid conv, 64 channels
    #   -> ReLU
    #   -> [B, 64, 568, 568]
    #
    # TODO 1-1: DoubleConv(nn.Module) 클래스를 직접 작성한다.
    # TODO 1-2: padding=0일 때 첫 conv 후 spatial size를 예상한다.
    # TODO 1-3: 두 번째 conv 후 spatial size를 예상한다.
    # TODO 1-4: DoubleConv(1, 64)를 만든 뒤 x를 통과시켜 shape을 출력한다.
    #
    # 확인 질문:
    # - 왜 572 -> 570 -> 568인가?
    # - channel 수가 1 -> 64로 바뀌는 것은 spatial size 변화와 어떤 관계인가?

    # ------------------------------------------------------------
    # 2. Encoder 한 단계 만들기
    # ------------------------------------------------------------
    # 목표 흐름:
    # DoubleConv output
    #   -> skip feature로 저장
    #   -> 2x2 MaxPool(stride=2)
    #   -> 다음 encoder 입력
    #
    # TODO 2-1: 2x2 MaxPool을 만든다.
    # TODO 2-2: 568x568 feature에 pooling을 적용했을 때 shape을 먼저 예상한다.
    # TODO 2-3: pooling 전 feature와 pooling 후 feature를 모두 출력한다.
    #
    # 확인 질문:
    # - skip connection에 저장해야 하는 것은 pooling 전인가, 후인가?
    # - 그 이유는 무엇이라고 생각하는가?

    # ------------------------------------------------------------
    # 3. Contracting path 확장
    # ------------------------------------------------------------
    # 논문 원형의 channel 흐름:
    # 1 -> 64 -> 128 -> 256 -> 512 -> 1024
    # spatial size는 valid conv와 max pooling 때문에 계속 감소한다.
    #
    # TODO 3-1: encoder block을 반복해서 4단계 contracting path를 만든다.
    # TODO 3-2: 각 단계의 pooling 전 feature를 e1, e2, e3, e4처럼 저장한다.
    # TODO 3-3: bottleneck까지 통과한 뒤 모든 feature shape을 출력한다.
    #
    # 반드시 출력할 것:
    # e1 shape
    # e2 shape
    # e3 shape
    # e4 shape
    # bottleneck shape
    #
    # 확인 질문:
    # - spatial resolution이 줄어들수록 feature가 담는 context는 어떻게 달라질까?

    # ------------------------------------------------------------
    # 4. Up-convolution만 먼저 확인
    # ------------------------------------------------------------
    # TODO 4-1: nn.ConvTranspose2d를 사용해 bottleneck feature를 2배 upsample한다.
    # TODO 4-2: upsample 전/후 shape을 출력한다.
    # TODO 4-3: 대응되는 encoder skip feature shape과 나란히 출력한다.
    #
    # 아직 concat하지 않는다.
    # 먼저 두 feature의 H, W가 같은지 직접 확인한다.

    # ------------------------------------------------------------
    # 5. Crop + Concatenate
    # ------------------------------------------------------------
    # 논문 U-Net은 valid convolution 때문에 encoder feature가 더 크다.
    # 따라서 encoder feature의 중앙을 crop해서 decoder feature와 H, W를 맞춘다.
    #
    # TODO 5-1: center crop 함수를 직접 작성한다.
    # TODO 5-2: skip feature를 decoder feature의 H, W에 맞춘다.
    # TODO 5-3: torch.cat(..., dim=1)로 channel 방향 concat을 수행한다.
    # TODO 5-4: concat 전/후 channel 수를 출력한다.
    #
    # 확인 질문:
    # - torch.cat(..., dim=1)에서 왜 dim=1인가?
    # - ResNet의 out + x와 무엇이 다른가?

    # ------------------------------------------------------------
    # 6. Decoder block 직접 구현
    # ------------------------------------------------------------
    # 목표:
    # up-conv -> crop -> concat -> DoubleConv
    #
    # TODO 6-1: Up block을 하나 만든다.
    # TODO 6-2: bottleneck + e4를 넣고 output shape을 확인한다.
    # TODO 6-3: 나머지 decoder stage를 순서대로 연결한다.

    # ------------------------------------------------------------
    # 7. 최종 1x1 convolution
    # ------------------------------------------------------------
    # 논문은 마지막에 1x1 convolution으로 각 pixel의 feature vector를
    # 원하는 class 수로 매핑한다.
    #
    # TODO 7-1: binary segmentation이라고 가정하고 output channel=1로 만든다.
    # TODO 7-2: 최종 logits shape을 출력한다.
    #
    # 주의:
    # 여기서는 sigmoid를 모델 내부에 넣지 않는다.
    # segmentation.py에서 BCEWithLogitsLoss와 연결하며 이유를 확인한다.

    # ------------------------------------------------------------
    # 8. 마지막 점검
    # ------------------------------------------------------------
    # 다음을 자신의 말로 설명할 수 있으면 이 파일의 목표를 달성한 것이다.
    # - contracting path의 역할
    # - bottleneck의 역할
    # - expansive path의 역할
    # - skip connection이 전달하는 정보
    # - crop이 필요한 이유
    # - concat 이후 channel 수가 변하는 이유


if __name__ == "__main__":
    main()
