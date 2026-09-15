"""
04-2. U-Net segmentation 실습

목표
- U-Net을 실제 segmentation 데이터 흐름과 연결한다.
- raw image / GT mask / model output / loss 관계를 직접 확인한다.
- 처음에는 synthetic binary segmentation으로 작은 문제를 만든다.
- 학습이 되면 prediction, IoU, Dice, failure case를 관찰한다.

핵심 질문
1. classification과 segmentation에서 GT의 shape은 어떻게 다른가?
2. U-Net output의 각 pixel 값은 무엇을 의미하는가?
3. BCEWithLogitsLoss를 사용할 때 sigmoid를 모델 안에 넣지 않는 이유는 무엇인가?
4. loss가 내려가도 segmentation이 나쁠 수 있는 경우는 무엇인가?
5. IoU와 Dice는 pixel accuracy와 무엇이 다른가?

진행 규칙
- unet_architecture.py의 구조를 먼저 이해한 뒤 진행한다.
- 실제 biomedical dataset보다 먼저 synthetic data로 pipeline을 검증한다.
- 한 번에 augmentation, weighted loss, elastic deformation까지 넣지 않는다.
"""

import torch


def main():
    torch.manual_seed(0)

    # ------------------------------------------------------------
    # 0. 이번 파일에서 확인할 전체 데이터 흐름
    # ------------------------------------------------------------
    # image
    #   -> Dataset / DataLoader
    #   -> U-Net
    #   -> logits
    #   -> BCEWithLogitsLoss(logits, GT mask)
    #   -> backward / optimizer step
    #   -> sigmoid
    #   -> binary prediction
    #   -> IoU / Dice

    # ------------------------------------------------------------
    # 1. Synthetic segmentation sample 만들기
    # ------------------------------------------------------------
    # 첫 실습에서는 검은 배경 위에 간단한 원 또는 사각형을 만든다.
    # image와 mask가 정확히 대응되는지 직접 눈으로 확인하는 것이 목적이다.
    #
    # TODO 1-1: 128x128 single-channel image tensor를 만든다.
    # TODO 1-2: 같은 크기의 binary GT mask를 만든다.
    # TODO 1-3: mask가 1인 영역의 image intensity가 다르게 보이도록 만든다.
    # TODO 1-4: image / mask의 shape, dtype, min, max를 출력한다.
    #
    # 예상 shape을 먼저 적을 것:
    # image: ?
    # mask:  ?

    # ------------------------------------------------------------
    # 2. Dataset으로 여러 sample 만들기
    # ------------------------------------------------------------
    # TODO 2-1: torch.utils.data.Dataset을 상속한 작은 dataset을 만든다.
    # TODO 2-2: sample마다 물체 위치/크기를 조금씩 다르게 만든다.
    # TODO 2-3: __getitem__이 (image, mask)를 반환하도록 한다.
    # TODO 2-4: DataLoader에서 batch 하나를 꺼내 shape을 출력한다.
    #
    # 확인 질문:
    # - classification label [B]와 segmentation mask [B, 1, H, W]는 무엇이 다른가?

    # ------------------------------------------------------------
    # 3. 작은 U-Net 연결
    # ------------------------------------------------------------
    # 처음에는 논문 원형보다 작은 channel 수를 사용해도 된다.
    # 예: 1 -> 16 -> 32 -> 64 -> ...
    # 목적은 논문 성능 재현이 아니라 데이터 흐름 확인이다.
    #
    # TODO 3-1: unet_architecture.py에서 직접 만든 구조를 가져오거나 다시 작성한다.
    # TODO 3-2: batch를 넣고 logits shape을 출력한다.
    # TODO 3-3: GT mask와 spatial size가 맞는지 확인한다.
    #
    # 논문 원형의 valid convolution을 그대로 쓰면 output이 input보다 작아진다.
    # 이 경우 GT mask도 output spatial size에 맞게 crop해야 한다.
    # 첫 학습 실험에서는 이 차이를 직접 확인한다.

    # ------------------------------------------------------------
    # 4. Loss 연결
    # ------------------------------------------------------------
    # TODO 4-1: binary segmentation용 BCEWithLogitsLoss를 만든다.
    # TODO 4-2: logits와 GT mask를 넣어 loss 하나를 계산한다.
    # TODO 4-3: loss 값과 tensor shape을 출력한다.
    #
    # 확인 질문:
    # - logits에는 왜 0~1 범위 제한이 없는가?
    # - sigmoid는 언제 적용해야 하는가?

    # ------------------------------------------------------------
    # 5. 한 번의 backward 확인
    # ------------------------------------------------------------
    # TODO 5-1: optimizer를 만든다.
    # TODO 5-2: zero_grad -> forward -> loss -> backward -> step 순서를 작성한다.
    # TODO 5-3: 첫 convolution weight의 gradient가 None이 아닌지 확인한다.
    #
    # 아직 epoch loop를 만들지 않는다.
    # 먼저 한 step의 학습 흐름이 완전히 연결되는지 확인한다.

    # ------------------------------------------------------------
    # 6. 작은 overfit 실험
    # ------------------------------------------------------------
    # 전체 dataset 학습 전에 sample 1~4개에 일부러 overfit해본다.
    # 이것은 segmentation pipeline이 제대로 연결되었는지 확인하는 최소 검증 실험이다.
    #
    # TODO 6-1: 아주 작은 subset만 사용한다.
    # TODO 6-2: 여러 step 학습하며 loss를 출력한다.
    # TODO 6-3: prediction이 GT에 가까워지는지 확인한다.
    #
    # 성공 기준:
    # - loss가 충분히 감소한다.
    # - 같은 sample에 대해 prediction mask가 GT 형태를 따라간다.

    # ------------------------------------------------------------
    # 7. 전체 synthetic dataset 학습
    # ------------------------------------------------------------
    # TODO 7-1: train / validation dataset을 분리한다.
    # TODO 7-2: epoch loop를 작성한다.
    # TODO 7-3: train loss와 validation loss를 기록한다.
    # TODO 7-4: 필요하면 outputs/04_unet/ 아래에 curve를 저장한다.

    # ------------------------------------------------------------
    # 8. Prediction 만들기
    # ------------------------------------------------------------
    # TODO 8-1: model.eval() + torch.no_grad()로 inference한다.
    # TODO 8-2: sigmoid(logits)로 foreground probability를 만든다.
    # TODO 8-3: threshold=0.5로 binary mask를 만든다.
    # TODO 8-4: input / GT / probability / prediction을 비교한다.

    # ------------------------------------------------------------
    # 9. IoU / Dice 직접 계산
    # ------------------------------------------------------------
    # TODO 9-1: intersection과 union을 tensor 연산으로 직접 구한다.
    # TODO 9-2: IoU를 계산한다.
    # TODO 9-3: Dice coefficient를 계산한다.
    # TODO 9-4: metric 식에서 epsilon이 왜 필요한지 확인한다.

    # ------------------------------------------------------------
    # 10. Failure case 관찰
    # ------------------------------------------------------------
    # TODO 10-1: validation sample 중 IoU가 낮은 예시를 찾는다.
    # TODO 10-2: input / GT / prediction / error map을 비교한다.
    # TODO 10-3: 다음 중 어떤 failure인지 추정한다.
    # - boundary가 흐림
    # - 작은 물체 누락
    # - background false positive
    # - 물체 내부 hole
    #
    # 아직 원인이라고 단정하지 말고 관찰과 가설을 분리한다.

    # ------------------------------------------------------------
    # 11. 논문 확장 실험 - 이후 진행
    # ------------------------------------------------------------
    # 기본 pipeline이 이해된 뒤 별도 실험으로 추가한다.
    # - elastic deformation augmentation
    # - touching object용 weighted loss
    # - 논문 원형 valid convolution vs padding=1 구현 비교
    # - encoder / decoder feature map visualization
    #
    # 한 번에 하나씩 조건을 바꿔 효과를 확인한다.


if __name__ == "__main__":
    main()
