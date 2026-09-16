# 05. DINOv2 — Learning Robust Visual Features without Supervision

재현 수준: **Official pretrained feature analysis**.

논문의 pretraining은 curated **LVD-142M**과 ViT-S/B/L/g, DINO+iBOT objectives, teacher EMA, KoLeo 등을 사용한다. 이를 개인 환경에서 재학습하는 것은 현실적이지 않으므로 **Meta 공식 DINOv2 checkpoint**를 사용한다.

분석 dataset은 논문 Table 8의 frozen-feature transfer benchmark 중 하나인 **Oxford-IIIT Pets**다. 즉 임의 dataset이 아니라 논문 평가에 실제 등장하는 dataset을 사용한다. 논문은 DINOv2 frozen features를 linear evaluation, retrieval, segmentation/depth 및 PCA patch visualization으로 평가한다.

## 파일
- `common.py`: official torch.hub DINOv2 + preprocessing
- `01_data.py`: Oxford-IIIT Pets 실제 데이터
- `02_features.py`: CLS / patch-token shape
- `03_pca.py`: 논문 qualitative analysis와 같은 patch-feature PCA
- `04_probe.py`: frozen CLS feature + linear probe
- `05_analyze.py`: frozen feature nearest-neighbor retrieval

```bash
python practice/05_dinov2/01_data.py
python practice/05_dinov2/02_features.py
python practice/05_dinov2/03_pca.py
python practice/05_dinov2/04_probe.py
python practice/05_dinov2/05_analyze.py
```

중요: 이 실습은 DINOv2 **training reproduction이 아니라 paper evaluation claim을 official pretrained features로 검증하는 분석**이다.
