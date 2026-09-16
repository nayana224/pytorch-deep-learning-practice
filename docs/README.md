# 논문 실습 문서

이 폴더는 `practice/`를 어떤 순서로 준비하고 실행할지 정리한 문서다.

## 읽는 순서

1. `01_DATA_SETUP.md` — 데이터와 checkpoint를 먼저 준비한다.
2. `02_STUDY_ORDER.md` — 논문별 코드 읽기/실행 순서를 따른다.
3. `03_PAPER_CHECKLIST.md` — 논문 1편을 끝냈다고 볼 기준을 점검한다.

## 기본 원칙

- 데이터 다운로드와 논문 공부 코드는 분리한다.
- `scripts/`는 다운로드/설정을 담당한다.
- `practice/`는 이미 준비된 데이터만 읽는다.
- 데이터가 이미 있으면 다시 다운로드하지 않는다.
- 큰 파일은 가능한 경우 이어받기(resume)를 사용한다.
- 논문에 실제 등장한 dataset / benchmark / task를 기본으로 사용한다.
- 논문 전체 재현이 현실적이지 않으면 `Faithful / Scaled / Pretrained analysis`를 구분한다.

## 전체 학습 순서

```text
환경 확인
→ 데이터 준비
→ 01 ResNet
→ 02 U-Net
→ 03 DeepLabv3+
→ 04 ViT
→ 05 DINOv2
→ 06 SAM
→ 07 Diffusion Policy
```

각 논문에서는 항상 아래 흐름으로 본다.

```text
paper problem
→ raw data
→ model code
→ tensor/feature shape
→ training or inference
→ visualization
→ failure case
→ paper claim과 비교
```
