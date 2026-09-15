# 02. U-Net — Convolutional Networks for Biomedical Image Segmentation

이 폴더는 U-Net 논문의 실제 실험 흐름을 PyTorch로 따라간다.

## 논문 기준
첫 실습은 논문의 EM segmentation 실험을 기준으로 한다.
- ISBI 2012 EM segmentation challenge
- training: 30 images
- image size: 512x512
- 각 image에 pixel-level GT segmentation map 존재
- original U-Net은 valid 3x3 convolution을 사용
- Figure 1의 input tile 572x572 → output 388x388
- training에서 pixel-wise softmax + cross entropy를 사용
- touching object를 강조하는 weight map과 elastic deformation은 기본 흐름을 이해한 뒤 추가

주의: dataset image 512x512와 Figure 1의 572x572 input tile은 같은 개념이 아니다.

## 진행 순서
1. `01_data.py`: 실제 EM image/GT를 읽고 shape, dtype, range, 의미 확인
2. `02_model.py`: Figure 1의 contracting/expanding path를 PyTorch로 직접 구현
3. `03_train.py`: image/GT/output/loss 연결, 작은 overfit sanity check 후 학습
4. `04_analyze.py`: prediction, IoU, error case, encoder/decoder feature 관찰
5. 이후 확장: elastic deformation → weighted loss → overlap-tile

코드는 대화에서 작은 단위로 받고 직접 타이핑한다. 미리 TODO scaffold를 채우는 방식은 사용하지 않는다.

## 완료 기준
1. Problem: sliding-window CNN의 중복 계산과 localization/context trade-off
2. Core idea: contracting path + symmetric expanding path + skip feature
3. Method: valid conv / pool / up-conv / crop / concat / 1x1 conv
4. Input / GT / Output / Loss: EM image → logits, segmentation map → pixel loss
5. Evidence: 논문 지표/결과와 직접 prediction 관찰
6. My observation: feature와 failure case에서 직접 본 현상
