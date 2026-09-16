# 04. Vision Transformer — An Image Is Worth 16x16 Words

재현 수준: **Scaled downstream training / paper architecture analysis**.

논문은 ImageNet, ImageNet-21k, JFT-300M으로 pretrain하고 ImageNet, CIFAR-10/100, Oxford-IIIT Pets, Flowers-102, VTAB으로 transfer한다. 로컬 기본 dataset은 논문에 실제 등장하는 **CIFAR-100**이다. JFT-300M/ImageNet-21k pretraining 자체는 현실적으로 재현하지 않으며 결과를 논문과 동일 성능 재현으로 해석하지 않는다.

## 데이터 준비
```bash
python scripts/download_torchvision_data.py cifar100
```
이미 있으면 `[skip]`하고 다시 받지 않는다. `practice/04_vit/` 코드는 자동 다운로드하지 않는다.

`vit.py`에는 Table 1의 **ViT-B/16: 12 layers, D=768, MLP=3072, 12 heads**를 구현했다. 논문 fine-tuning 결과는 384 resolution을 사용하므로 base path는 384로 둔다. 로컬 sanity training용 `vit_tiny16`도 같은 patch→CLS→position→Transformer data flow를 유지한다.

## 파일
- `01_data.py`: CIFAR-100 실제 image/label, 32→384 확인
- `02_patches.py`: 16×16 patches를 실제로 펼쳐 시각화
- `vit.py`: patch embedding, CLS token, positional embedding, Transformer encoder
- `03_model.py`: token/attention shape trace
- `04_train.py`: CIFAR-100 scaled training (`--model base`는 무거움)
- `05_analyze.py`: CLS→patch attention overlay

```bash
python practice/04_vit/01_data.py
python practice/04_vit/02_patches.py
python practice/04_vit/03_model.py
python practice/04_vit/04_train.py --model tiny
python practice/04_vit/05_analyze.py
```
