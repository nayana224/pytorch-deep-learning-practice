"""
04-2. U-Net segmentation typing practice

unet_architecture.py에서 네트워크 구조를 직접 타이핑해 이해한 뒤 진행한다.
이 파일도 TODO 목록을 미리 채워두지 않는다.
ChatGPT가 대화에서 실제 코드를 작은 단위로 제시하면 사용자가 직접 타이핑하며 누적한다.

이 파일에서 최종적으로 확인할 흐름
raw image / GT mask
    -> Dataset / DataLoader
    -> U-Net
    -> logits
    -> loss
    -> backward / optimizer step
    -> probability / prediction
    -> IoU / Dice
    -> failure case visualization

처음에는 실제 biomedical dataset보다 synthetic binary segmentation으로
데이터 파이프라인이 올바르게 연결되는지 검증한다.
이후 elastic deformation, weighted loss, 실제 데이터셋 등은 한 번에 하나씩 추가한다.
"""

import torch


def main():
    torch.manual_seed(0)

    print("U-Net architecture practice를 먼저 완료한 뒤 이 파일을 진행합니다.")


if __name__ == "__main__":
    main()
