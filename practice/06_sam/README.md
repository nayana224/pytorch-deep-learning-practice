# 06. SAM — Segment Anything

이 폴더는 Segment Anything 논문의 promptable segmentation 구조와 zero-shot transfer를 직접 관찰한다.

## 논문 기준
논문은 SA-1B(11M images, 1.1B masks)와 data engine을 사용한다. 모델은:
- image encoder
- prompt encoder
- lightweight mask decoder
로 구성되며 point / box / mask prompt를 처리한다.

SA-1B 전체 학습 재현은 개인 실습 규모에서 현실적이지 않다. 따라서 공식 pretrained SAM을 사용해 논문의 핵심 동작을 검증한다.

## 진행 순서
1. `01_image.py`: 실제 이미지 입력과 preprocessing 확인
2. `02_point_prompt.py`: point prompt → masks / scores
3. `03_box_prompt.py`: box prompt → mask 비교
4. `04_ambiguity.py`: 하나의 ambiguous prompt에서 multiple masks 관찰
5. `05_analyze.py`: prompt 변화, boundary, small object, failure case 분석

가능하면 논문에서 사용한 zero-shot segmentation 성격에 맞춰 새로운 이미지 분포에서도 prompt를 바꿔 결과를 본다.

## 완료 기준
1. Problem: 고정 task용 segmentation을 넘어 prompt로 새로운 segmentation 문제를 풀 수 있는가
2. Core idea: promptable segmentation + large-scale mask data engine
3. Method: image encoder / prompt encoder / mask decoder / multiple masks
4. Input / GT / Output / Loss: image + prompt → masks/scores; training에서는 focal + dice 계열 mask supervision
5. Evidence: zero-shot downstream 결과와 prompt experiment
6. My observation: prompt 변화와 ambiguity/failure case에서 직접 본 현상
