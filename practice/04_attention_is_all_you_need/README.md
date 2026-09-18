# 04. Attention Is All You Need

실습 깊이: **Level 2 — Core mechanism check**

이 폴더의 목적은 Transformer 번역 모델 전체를 재현하는 것이 아니다.  
논문의 핵심 메커니즘을 작은 tensor로 직접 계산하고, 결과를 그림으로 확인한다.


## 공통 첫 바퀴 실행

이 폴더의 핵심 실습만 연속 실행하려면:

```bash
python practice/04_attention_is_all_you_need/00_run_core.py
```

전체 training을 자동으로 수행하는 명령이 아니라, 첫 바퀴에서 봐야 할 핵심 메커니즘만 실행한다.
생성된 그림은 `outputs/04_attention_is_all_you_need/`에서 확인한다.

## Paper claim

Transformer는 recurrence와 convolution 없이도 attention만으로 sequence 관계를 모델링할 수 있다.

첫 바퀴에서는 이 주장을 전부 재현하지 않고, 아래 핵심 구성요소가 실제 tensor에서 어떻게 동작하는지만 확인한다.

## 반드시 확인할 것

1. 같은 token embedding이 서로 다른 Q / K / V projection으로 바뀐다.
2. `QK^T`가 Query-Key 관계 score를 만든다.
3. `sqrt(d_k)`로 scaling한 뒤 softmax하면 attention weight가 된다.
4. decoder causal mask는 미래 token의 attention weight를 0으로 만든다.
5. 서로 다른 head는 다른 projection을 사용해 다른 attention pattern을 만들 수 있다.
6. positional encoding이 순서 정보가 없는 attention 입력에 위치 정보를 더한다.
7. cross-attention에서는 **Q는 decoder**, **K/V는 encoder output**에서 나온다.

## 파일

- `01_qkv.py`: Q/K/V projection과 shape
- `02_scaled_dot_product.py`: 논문 attention 수식을 직접 계산
- `03_masked_attention.py`: unmasked vs causal masked attention
- `04_multihead.py`: head별 attention map 비교
- `05_positional_encoding.py`: sinusoidal positional encoding
- `06_cross_attention.py`: encoder-decoder cross-attention
- `00_run_core.py`: 첫 바퀴 핵심 실습 전체 실행

## 실행

```bash
python practice/04_attention_is_all_you_need/00_run_core.py
```

## 생성 결과

```text
outputs/04_attention_is_all_you_need/
├── 01_qkv_projection.png
├── 02_scaled_dot_product_attention.png
├── 03_masked_vs_unmasked.png
├── 04_multihead_attention.png
├── 05_positional_encoding.png
└── 06_cross_attention.png
```

## What to observe

- 한 Query row가 attention matrix에서 무엇을 뜻하는가?
- softmax 이후 각 row의 값은 왜 합이 1인가?
- mask를 적용하면 왜 미래 token의 weight가 0이 되는가?
- head마다 attention pattern이 달라지는 이유는 무엇인가?
- positional encoding이 없으면 token 순서를 어떻게 구분할 것인가?
- self-attention과 cross-attention에서 Q/K/V의 출처가 어떻게 다른가?

## 이 실습에서 하지 않는 것

- WMT 번역 dataset 전처리
- tokenizer 학습
- encoder/decoder 전체 stack 학습
- teacher forcing 기반 번역 학습
- BLEU 재현
- 논문의 모든 ablation 재현

이 폴더는 **Transformer 전체 구현 능력**이 아니라 **attention 메커니즘 이해**를 확인하기 위한 실습이다.
