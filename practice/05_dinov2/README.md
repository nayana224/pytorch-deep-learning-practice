# 05. DINOv2 — Learning Robust Visual Features without Supervision

이 폴더는 DINOv2 논문의 핵심 주장인 general-purpose frozen visual feature를 직접 관찰하는 실습이다.

## 논문 기준
논문은 자동 curated LVD-142M 데이터셋과 대규모 ViT self-supervised pre-training을 사용한다. 핵심 학습 신호는 image-level DINO objective, patch-level iBOT objective, teacher EMA, centering/Sinkhorn-Knopp, KoLeo regularizer 등이다.

전체 LVD-142M pre-training은 개인 실습 규모에서 현실적이지 않다. 따라서 이 폴더에서는:
- 공식 pretrained DINOv2 모델을 우선 사용
- 논문에서 주장한 image-level / pixel-level feature 성질을 직접 관찰
- PCA feature visualization, nearest-neighbor/retrieval, linear probe 또는 segmentation/depth용 frozen feature 분석
을 중심으로 한다.

논문 학습 재현이 아니라면 `pretrained feature analysis`라고 명확히 구분한다.

## 진행 순서
1. `01_data.py`: 분석할 실제 이미지 데이터 준비 및 preprocessing 확인
2. `02_features.py`: official pretrained model에서 class/patch feature 추출
3. `03_pca.py`: 논문 Figure 1과 연결해 patch PCA visualization
4. `04_probe.py`: frozen feature로 간단한 linear probe 또는 nearest-neighbor 평가
5. `05_analyze.py`: 배경/물체/부분 대응과 failure case 관찰

## 완료 기준
1. Problem: label/text 없이도 범용 visual feature를 만들 수 있는가
2. Core idea: curated large-scale data + discriminative self-supervised objectives
3. Method: student/teacher, DINO + iBOT, ViT, EMA
4. Input / GT / Output / Loss: raw image crops / masked patches → teacher-student feature objectives
5. Evidence: frozen feature의 image/pixel-level benchmark 성능
6. My observation: PCA/nearest-neighbor/feature 대응에서 직접 본 현상
