# 03. 논문 1편 완료 체크리스트

논문을 읽었다는 기준은 PDF를 끝까지 넘긴 것이 아니라, 아래 내용을 자신의 말로 설명하고 최소 한 번의 실습으로 확인한 상태다.

## 6가지 완료 기준

### 1. Problem

기존 방법에 어떤 문제가 있었는가?

확인 질문:

```text
이 논문이 없었다면 어떤 문제가 남아 있었는가?
저자들은 기존 방법의 어떤 한계를 직접 지적했는가?
```

### 2. Core idea

저자들이 그 문제를 어떤 핵심 아이디어로 풀었는가?

확인 질문:

```text
논문의 핵심을 한두 문장으로 줄이면 무엇인가?
왜 이 아이디어가 기존 문제에 맞는가?
```

### 3. Method

그 아이디어를 실제 모델 구조로 어떻게 구현했는가?

확인 항목:

```text
주요 block
channel / token / feature 흐름
skip / attention / decoder / diffusion 등 핵심 연산
논문 Figure와 코드의 대응
```

### 4. Input / GT / Output / Loss

학습 데이터가 모델 안에서 어떻게 흐르는가?

반드시 확인:

```text
Input shape / dtype / range
GT shape / dtype / 의미
Model output shape / 의미
Loss가 어떤 두 값을 비교하는가
```

### 5. Evidence

실험 결과가 저자 주장을 실제로 뒷받침하는가?

확인 질문:

```text
어떤 baseline과 비교했는가?
어떤 metric을 사용했는가?
어떤 ablation이 핵심 아이디어를 검증하는가?
결과가 주장보다 약하거나 애매한 부분은 없는가?
```

### 6. My observation

직접 실습했을 때 무엇을 봤는가?

최소 하나 이상 남긴다.

```text
feature map
attention map
prediction
probability
error map
failure case
retrieval result
action trajectory
```

## 논문 필기 노트 템플릿

```text
[논문명]

1. 이 논문이 해결하려는 문제
2. 기존 방법의 한계
3. 핵심 아이디어
4. 모델 구조
5. Input / GT / Output / Loss
6. 핵심 실험 결과
7. 내가 직접 한 실습
8. Feature / Prediction Visualization
9. 틀린 사례와 원인 추정
10. 이 논문에서 내가 가져갈 한 문장
```

## 완료 판정

아래에 모두 답할 수 있으면 다음 논문으로 넘어간다.

```text
[ ] Problem을 내 말로 설명할 수 있다.
[ ] Core idea를 내 말로 설명할 수 있다.
[ ] 논문 Figure를 보며 model code 위치를 찾을 수 있다.
[ ] Input / GT / Output / Loss shape와 의미를 설명할 수 있다.
[ ] 핵심 실험 하나 이상이 무엇을 검증하는지 설명할 수 있다.
[ ] 직접 실행한 결과나 시각화 하나 이상을 설명할 수 있다.
[ ] failure case 또는 한계 하나 이상을 말할 수 있다.
[ ] 마지막으로 논문에서 가져갈 한 문장을 정리했다.
```
