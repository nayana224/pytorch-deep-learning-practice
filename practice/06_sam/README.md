# 06. SAM — Segment Anything

실습 깊이: **Level 3**

재현 수준: **Official pretrained SAM + SA-1B subset analysis**

논문은 promptable segmentation task, SAM(image encoder + prompt encoder + lightweight mask decoder), 그리고 **SA-1B: 11M images / 1.1B masks**를 함께 제안한다. 전체 SA-1B를 재학습하는 대신 official SAM checkpoint를 사용하되, 실습 image/GT도 논문 데이터인 **SA-1B의 실제 shard/subset**을 사용한다.

SA-1B는 research license 동의가 필요한 dataset이므로 자동 다운로드하지 않는다. Meta 공식 SA-1B 페이지에서 라이선스에 동의한 뒤 일부 shard를 `data/06_sam/sa1b/`에 둔다.


## 공통 첫 바퀴 실행

이 폴더의 핵심 실습만 연속 실행하려면:

```bash
python practice/06_sam/00_run_core.py
```

전체 training을 자동으로 수행하는 명령이 아니라, 첫 바퀴에서 봐야 할 핵심 메커니즘만 실행한다.
생성된 그림은 `outputs/06_sam/`에서 확인한다.

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
- `04_ambiguity.py`: single-point **3 multimask outputs**와 IoU ranking
- `05_analyze.py`: automatic grid-prompt 계열 mask generation 관찰

이 폴더는 training reproduction이 아니라 pretrained promptable-segmentation behavior 분석이다.
