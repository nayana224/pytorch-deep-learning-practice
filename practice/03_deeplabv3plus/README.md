# 03. DeepLabv3+ — Encoder-Decoder with Atrous Separable Convolution

이 폴더는 DeepLabv3+ 논문의 실제 segmentation 설정을 기준으로 공부한다.

## 논문 기준
논문은 PASCAL VOC 2012와 Cityscapes에서 평가한다. 실습은 먼저 PASCAL VOC 2012로 진행하고, 이후 Cityscapes로 확장한다.

핵심 요소:
- semantic segmentation
- encoder: DeepLabv3 / ASPP
- atrous convolution과 output stride
- decoder에서 low-level feature 결합
- low-level channel을 1x1 conv로 줄인 뒤 concat
- depthwise separable convolution
- metric: mIoU

논문은 VOC 2012에서 train/val/test와 추가 annotation(trainaug)을 사용하고, Cityscapes에서도 결과를 보고한다.

## 진행 순서
1. `01_data.py`: VOC image / segmentation mask / class index / transform 확인
2. `02_atrous.py`: standard conv와 atrous conv의 receptive field/shape 비교
3. `03_model.py`: ASPP + decoder 데이터 흐름 구현
4. `04_train.py`: image → logits → pixel CE loss → mIoU
5. `05_analyze.py`: boundary, small object, low-level feature, output stride 비교

## 재현 원칙
논문의 전체 Xception/ResNet-101 training을 처음부터 재현하는 것은 계산비용이 크다. 데이터/metric/구조는 논문에 맞추고, backbone 크기나 학습 epoch를 줄이면 README와 결과에 명시한다.

## 완료 기준
1. Problem: semantic context와 sharp boundary를 동시에 얻기 어려움
2. Core idea: ASPP encoder + decoder 결합
3. Method: atrous conv / ASPP / low-level feature / separable conv
4. Input / GT / Output / Loss: RGB image → class logits, pixel class mask → CE
5. Evidence: mIoU와 decoder/output-stride 비교
6. My observation: boundary와 failure case에서 직접 본 차이
