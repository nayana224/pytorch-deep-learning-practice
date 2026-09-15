# 04. U-Net 실습

이 폴더의 목표는 U-Net 완성 코드를 복사하는 것이 아니라, 논문에서 본 구조를 PyTorch tensor 연산으로 직접 확인하는 것이다.

## 실습 순서

### 1. `unet_architecture.py`
네트워크 구조만 다룬다.

진행 순서:
1. 입력 tensor shape 확인
2. `DoubleConv` 직접 구현
3. max pooling으로 encoder 한 단계 확인
4. contracting path 확장
5. `ConvTranspose2d`로 up-convolution 확인
6. encoder feature center crop
7. `torch.cat(..., dim=1)`로 skip feature 결합
8. decoder block 연결
9. 마지막 `1x1 Conv`로 segmentation logits 생성

각 단계에서 **코드를 쓰기 전에 예상 shape을 먼저 적고**, 실행 결과와 비교한다.

첫 시작점은 `DoubleConv` 하나다. 전체 U-Net을 한 번에 구현하지 않는다.

## 2. `segmentation.py`
구조를 실제 segmentation 학습 흐름과 연결한다.

진행 순서:
1. synthetic image / GT mask 생성
2. `Dataset` / `DataLoader`에서 batch shape 확인
3. U-Net forward와 logits shape 확인
4. `BCEWithLogitsLoss` 연결
5. 한 번의 backward 확인
6. 1~4개 sample에 overfit하는 최소 검증 실험
7. 전체 synthetic dataset 학습
8. sigmoid + threshold로 prediction 생성
9. IoU / Dice 직접 계산
10. 낮은 IoU의 failure case 관찰

## 논문과 연결해서 볼 질문

- contracting path는 무엇을 잃고 무엇을 얻는가?
- expansive path만으로 원래 위치 정보를 완벽하게 복원할 수 있는가?
- 그래서 encoder의 high-resolution feature를 왜 decoder에 전달하는가?
- ResNet의 `F(x) + x`와 U-Net의 `torch.cat([skip, up], dim=1)`은 어떻게 다른가?
- original U-Net에서 valid convolution 때문에 왜 crop이 필요한가?
- 마지막 `1x1 Conv`는 각 pixel에서 무엇을 계산하는가?

## 기본 실습 이후에만 추가할 것

기본 데이터 흐름을 확인하기 전에는 아래 항목을 동시에 넣지 않는다.

- elastic deformation
- touching-cell weighted loss
- 실제 biomedical dataset
- feature visualization
- padding=1 형태의 modern U-Net 변형

기본 실습이 끝난 뒤 하나씩 추가하면서 결과를 비교한다.

## 완료 기준

다음 내용을 직접 설명할 수 있으면 기본 U-Net 실습을 완료한 것으로 본다.

1. Problem: sliding-window CNN의 비효율과 localization/context trade-off
2. Core idea: contracting path + expanding path + skip feature
3. Method: downsampling / upsampling / crop / concat / 1x1 convolution
4. Input / GT / Output / Loss: image → logits, mask → loss의 흐름
5. Evidence: IoU / Dice 및 prediction 결과
6. My observation: encoder/decoder feature와 failure case에서 직접 본 현상
