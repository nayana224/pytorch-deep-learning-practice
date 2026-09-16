# 06. SAM — Segment Anything

재현 수준: **Official pretrained SAM + SA-1B subset analysis**.

논문은 promptable segmentation task, SAM(image encoder + prompt encoder + lightweight mask decoder), 그리고 **SA-1B: 11M images / 1.1B masks**를 함께 제안한다. 전체 SA-1B를 재학습하는 대신 official SAM checkpoint를 사용하되, 실습 image/GT도 논문 데이터인 **SA-1B의 실제 shard/subset**을 사용한다. 임의 COCO/인터넷 이미지를 기본 dataset으로 대체하지 않는다.

SA-1B는 research license 동의가 필요한 dataset이므로 자동 다운로드 스크립트로 우회하지 않는다. Meta 공식 SA-1B 페이지에서 라이선스에 동의한 뒤 일부 shard를 `data/06_sam/sa1b/`에 둔다. image 옆에 같은 stem의 JSON annotation이 있어야 한다.

## 준비
```bash
bash scripts/setup_sam.sh
bash scripts/download_sam_vit_b.sh
# SA-1B 공식 shard/subset을 data/06_sam/sa1b/ 아래에 배치
```

## 분석
```bash
python practice/06_sam/01_image.py
python practice/06_sam/02_point_prompt.py
python practice/06_sam/03_box_prompt.py
python practice/06_sam/04_ambiguity.py
python practice/06_sam/05_analyze.py
```

- `02_point_prompt.py`: SA-1B GT 내부 foreground point → mask / predicted-IoU / actual IoU
- `03_box_prompt.py`: SA-1B GT bbox → box-prompt segmentation
- `04_ambiguity.py`: 논문의 핵심인 single-point **3 multimask outputs**와 IoU ranking
- `05_analyze.py`: automatic grid-prompt 계열 mask generation을 관찰

논문 training loss는 focal + dice이며, ambiguity 학습에서는 multiple masks 중 minimum loss에 backprop하고 각 mask의 estimated IoU도 예측한다. 이 폴더는 training reproduction이 아니라 pretrained promptable-segmentation behavior 분석이다.
