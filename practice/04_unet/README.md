# 04. U-Net 실습

이 폴더의 목표는 U-Net 완성 코드를 복사하거나 TODO를 채우는 것이 아니라, **대화에서 받은 실제 코드를 사용자가 직접 타이핑하고 실행하면서 논문 구조를 이해하는 것**이다.

## 학습 방식

이 실습은 다음 순서를 반복한다.

1. ChatGPT가 작은 실행 단위의 실제 코드를 제시한다.
2. 사용자가 코드를 직접 타이핑한다.
3. 실행 전에 가능하면 tensor shape이나 동작을 예상한다.
4. 실행 결과를 확인한다.
5. 이해되지 않는 줄, shape 변화, 연산의 이유를 질문한다.
6. 이해가 끝나면 다음 코드 조각으로 넘어간다.

한 번에 전체 U-Net 코드를 받지 않는다. 최종적으로는 사용자가 직접 타이핑한 완성 코드가 파일에 남는다.

## 1. `unet_architecture.py`

논문 Figure 1의 original U-Net 구조를 PyTorch로 직접 따라간다.

최종적으로 다룰 흐름은 다음과 같다.

```text
input
→ valid 3x3 convolution + ReLU
→ valid 3x3 convolution + ReLU
→ max pooling
→ contracting path
→ bottleneck
→ up-convolution
→ encoder feature crop
→ concatenation
→ expanding path
→ 1x1 convolution
→ segmentation logits
```

첫 입력은 논문 Figure 1과 같은 `[1, 1, 572, 572]`에서 시작한다.

핵심 관찰 항목:
- `572 → 570 → 568`이 되는 이유
- spatial size가 줄고 channel 수가 증가하는 흐름
- pooling 전 encoder feature를 skip으로 저장하는 이유
- upsampled decoder feature와 encoder feature의 spatial mismatch
- valid convolution 때문에 crop이 필요한 이유
- `torch.cat(..., dim=1)` 전후 channel 변화
- ResNet의 addition과 U-Net의 concatenation 차이
- 마지막 `1x1 Conv`가 pixel별 feature를 class logits로 바꾸는 과정

## 2. `segmentation.py`

네트워크 구조를 이해한 뒤 실제 segmentation 학습 데이터 흐름을 연결한다.

최종적으로 확인할 흐름:

```text
raw image / GT mask
→ Dataset / DataLoader
→ U-Net
→ logits
→ loss
→ backward / optimizer step
→ probability / prediction
→ IoU / Dice
→ failure case visualization
```

처음에는 synthetic binary segmentation을 사용한다. pipeline이 검증된 뒤 실제 biomedical dataset, elastic deformation, touching-cell weighted loss 등을 한 번에 하나씩 추가한다.

## 논문과 연결해서 계속 질문할 것

- contracting path는 무엇을 잃고 무엇을 얻는가?
- expansive path만으로 localization을 충분히 복원하기 어려운 이유는 무엇인가?
- encoder의 high-resolution feature가 decoder에 어떤 정보를 보완하는가?
- original U-Net에서 valid convolution과 crop은 어떻게 연결되는가?
- 마지막 `1x1 Conv`는 각 pixel에서 정확히 무엇을 계산하는가?

## 완료 기준

다음 내용을 자신의 말로 설명하고 최소 한 번 직접 실행해 확인할 수 있으면 기본 U-Net 실습을 완료한 것으로 본다.

1. Problem: sliding-window CNN의 비효율과 localization/context trade-off
2. Core idea: contracting path + expanding path + skip feature
3. Method: downsampling / upsampling / crop / concat / 1x1 convolution
4. Input / GT / Output / Loss: image → logits, mask → loss의 흐름
5. Evidence: IoU / Dice 및 prediction 결과
6. My observation: encoder/decoder feature와 failure case에서 직접 본 현상
