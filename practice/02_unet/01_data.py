"""U-Net paper practice 01: inspect the original ISBI 2012 EM dataset.

진행 방식
- 먼저 scripts/download_isbi2012.sh 로 실제 challenge archive를 받는다.
- archive에서 풀린 TIFF 파일명을 find로 확인한다.
- 그 다음 대화에서 실제 코드를 작은 단위로 받아 이 파일에 직접 타이핑한다.

첫 목표
1. training image TIFF stack 읽기
2. GT TIFF stack 읽기
3. stack / slice shape, dtype, value range 확인
4. GT unique values와 class 의미 확인
5. 같은 index의 image / GT를 나란히 시각화
6. foreground / background pixel 비율 확인

아직 model, augmentation, loss는 작성하지 않는다.
"""
