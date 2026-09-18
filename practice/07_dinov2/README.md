# 07. DINOv2 — Learning Robust Visual Features without Supervision

실습 깊이: **Level 3**

재현 수준: **Official pretrained feature analysis**

논문의 pretraining은 curated **LVD-142M**과 ViT-S/B/L/g, DINO+iBOT objectives, teacher EMA, KoLeo 등을 사용한다. 이를 개인 환경에서 재학습하는 것은 현실적이지 않으므로 **Meta 공식 DINOv2 checkpoint**를 사용한다.

분석 dataset은 논문 Table 8의 frozen-feature transfer benchmark 중 하나인 **Oxford-IIIT Pets**다. 논문은 DINOv2 frozen features를 linear evaluation, retrieval, segmentation/depth 및 PCA patch visualization으로 평가한다.


## 공통 첫 바퀴 실행

이 폴더의 핵심 실습만 연속 실행하려면:

```bash
python practice/07_dinov2/00_run_core.py
```

전체 training을 자동으로 수행하는 명령이 아니라, 첫 바퀴에서 봐야 할 핵심 메커니즘만 실행한다.
생성된 그림은 `outputs/07_dinov2/`에서 확인한다.

## 준비

공식 DINOv2 repo와 pretrained ViT-S/14 weight를 먼저 준비한다.

```bash
bash scripts/setup_dinov2.sh
python scripts/download_torchvision_data.py pets
```

setup 이후 `practice/` 코드는 GitHub repo를 자동으로 다운로드하지 않는다.
이미 준비된 `external/dinov2`와 torch hub cache를 사용한다.

## 파일

- `common.py`: local official DINOv2 checkout + cached pretrained weight + preprocessing
- `01_data.py`: Oxford-IIIT Pets 실제 데이터
- `02_features.py`: CLS / patch-token shape
- `03_pca.py`: patch-feature PCA
- `04_probe.py`: frozen CLS feature + linear probe
- `05_analyze.py`: frozen feature nearest-neighbor retrieval

## 실행

```bash
python practice/07_dinov2/01_data.py
python practice/07_dinov2/02_features.py
python practice/07_dinov2/03_pca.py
python practice/07_dinov2/04_probe.py
python practice/07_dinov2/05_analyze.py
```

중요: 이 실습은 DINOv2 **training reproduction이 아니라 paper evaluation claim을 official pretrained features로 검증하는 분석**이다.
