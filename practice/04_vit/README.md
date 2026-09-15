# 04. Vision Transformer (ViT) — An Image Is Worth 16x16 Words

이 폴더는 ViT 논문의 데이터 흐름과 구조를 PyTorch로 따라간다.

## 논문 기준
논문은 ImageNet, ImageNet-21k, JFT-300M 등 대규모 pre-training을 사용하고 CIFAR-10/100, Oxford-IIIT Pets, Flowers, VTAB 등으로 transfer한다.

로컬 실습에서는 계산비용 때문에 JFT-300M pre-training을 재현하지 않는다. 대신 논문에 실제로 등장하는 공개 downstream dataset(CIFAR-100 또는 CIFAR-10)을 사용해 patch/token 흐름과 fine-tuning 구조를 검증한다. pretrained checkpoint를 사용할 경우 출처와 사전학습 데이터 차이를 명시한다.

핵심 요소:
- image → fixed-size patches
- flatten + linear projection
- class token
- position embedding
- Transformer encoder
- MSA + MLP + residual
- classification head

## 진행 순서
1. `01_data.py`: 실제 downstream dataset image/label 확인
2. `02_patches.py`: image를 patch sequence로 바꾸고 shape/시각화
3. `03_model.py`: patch embedding → class token → position → encoder
4. `04_train.py`: logits / CE loss / fine-tuning 또는 작은-scale training
5. `05_analyze.py`: attention map, patch size, failure case 관찰

## 완료 기준
1. Problem: CNN inductive bias 없이 Transformer를 vision에 직접 적용할 수 있는가
2. Core idea: image patch를 token처럼 다룸
3. Method: patch embedding + positional embedding + Transformer encoder
4. Input / GT / Output / Loss: image → class logits, class id → CE
5. Evidence: 논문 transfer 결과와 작은 실험
6. My observation: patch/attention/failure case에서 직접 본 현상
