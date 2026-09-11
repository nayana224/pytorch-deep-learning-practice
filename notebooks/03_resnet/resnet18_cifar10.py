"""
03-2. ResNet-18 — CIFAR-10 실습

목표
- CIFAR-10 기준으로 직접 만든 residual block을 전체 네트워크로 확장한다.
- Plain CNN과 ResNet의 학습 흐름을 같은 조건에서 비교한다.
- 완성 코드는 미리 넣지 않고, 아래 TODO를 직접 타이핑하며 채운다.

핵심 질문
1. CIFAR-10 입력 / GT / output / loss는 어떻게 연결되는가?
2. BasicBlock을 여러 개 쌓으면 전체 ResNet 구조가 어떻게 만들어지는가?
3. 같은 조건에서 Plain CNN과 ResNet의 training curve는 어떻게 다른가?
4. 예측 실패 사례에서 무엇을 관찰할 수 있는가?

진행 순서
Step 1. 환경 / seed / device 설정
Step 2. CIFAR-10 transform / DataLoader 구성
Step 3. 한 batch의 input / label shape 확인
Step 4. BasicBlock 직접 구현
Step 5. 작은 Plain CNN 구현
Step 6. 작은 ResNet 구현
Step 7. loss / optimizer 설정
Step 8. 공통 train / evaluate 함수 구현
Step 9. 동일 조건으로 Plain / ResNet 학습
Step 10. loss / accuracy 기록 및 비교
Step 11. prediction / failure case 확인

주의
- 한 번에 한 단계씩 구현한다.
- Plain / ResNet 비교 시 seed, batch size, optimizer, epoch 등 조건을 동일하게 유지한다.
- 단일 실행의 accuracy 차이만으로 일반적인 우수성을 결론내리지 않는다.
"""

import torch
import torch.nn as nn


def main():
    # Step 1 -------------------------------------------------------------
    # TODO: random seed와 device를 설정한다.
    pass

    # Step 2 -------------------------------------------------------------
    # TODO: CIFAR-10 transform과 DataLoader를 구성한다.

    # Step 3 -------------------------------------------------------------
    # TODO: 한 batch를 꺼내 input / label의 shape, dtype, range를 확인한다.

    # Step 4 -------------------------------------------------------------
    # TODO: BasicBlock을 직접 구현한다.
    # - residual branch
    # - identity shortcut
    # - 필요 시 projection shortcut

    # Step 5 -------------------------------------------------------------
    # TODO: 비교용 작은 Plain CNN을 구현한다.

    # Step 6 -------------------------------------------------------------
    # TODO: 직접 만든 BasicBlock을 사용해 작은 ResNet을 구현한다.

    # Step 7 -------------------------------------------------------------
    # TODO: loss function과 optimizer를 설정한다.

    # Step 8 -------------------------------------------------------------
    # TODO: Plain / ResNet에 공통으로 사용할 train / evaluate 함수를 구현한다.

    # Step 9 -------------------------------------------------------------
    # TODO: 두 모델을 같은 조건으로 학습한다.

    # Step 10 ------------------------------------------------------------
    # TODO: epoch별 loss / accuracy를 저장하고 비교한다.

    # Step 11 ------------------------------------------------------------
    # TODO: prediction 예시와 failure case를 확인한다.


if __name__ == "__main__":
    main()
