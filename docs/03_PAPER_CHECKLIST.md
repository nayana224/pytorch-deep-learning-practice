# 03. 논문 완료 체크리스트

논문을 읽었다는 기준과 실습을 깊게 했다는 기준을 분리한다.

## Level 1 — 논문을 읽었다

아래 5가지를 내 말로 설명할 수 있으면 된다.

1. **Problem** — 기존 방법의 문제가 무엇인가?
2. **Core idea** — 저자는 어떤 핵심 아이디어를 제안했는가?
3. **Method** — 그 아이디어를 구조로 어떻게 구현했는가?
4. **Input / GT / Output / Loss** — 데이터가 모델 안에서 어떻게 흐르는가?
5. **Evidence** — 어떤 실험이 저자의 주장을 뒷받침하는가?

## Level 2 — 핵심 메커니즘을 직접 봤다

논문 전체를 구현할 필요는 없다.

다음 중 논문에 가장 중요한 것 하나 이상을 직접 확인한다.

```text
residual addition
crop + concat
atrous sampling
attention matrix
patch token
prompt -> mask
feature PCA
action chunk
forward noising
action denoising
```

결과는 `outputs/<paper>/`에 남긴다.

## Level 3 — 실제 모델 행동을 봤다

연구와 직접 연결되는 논문에서만 수행한다.

```text
actual dataset
official/pretrained/faithful model
prediction
feature
metric
failure case
```

## 필기 노트

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

Level 1만 진행한 시점에는 7~9번을 비워도 된다.  
Level 2/3 실습을 진행할 때 채운다.
