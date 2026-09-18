# 04. Attention Is All You Need

실습 깊이: **Level 2 — Core mechanism check**

이 폴더의 목적은 Transformer 번역 모델 전체를 재현하는 것이 아니다.  
논문의 핵심 메커니즘인 **Q/K/V → scaled dot-product attention → mask → multi-head** 흐름을 작은 tensor로 직접 확인한다.

## 반드시 확인할 것

1. 같은 token embedding이 서로 다른 Q / K / V projection으로 바뀐다.
2. `QK^T`가 token 간 유사도/관계 score를 만든다.
3. `sqrt(d_k)`로 scaling한 뒤 softmax하면 attention weight가 된다.
4. decoder causal mask는 미래 token의 attention weight를 0으로 만든다.
5. 서로 다른 head는 다른 projection을 사용하므로 다른 attention pattern을 만들 수 있다.

## 파일

- `01_qkv.py`: Q/K/V projection과 shape
- `02_scaled_dot_product.py`: 논문 attention 수식을 직접 계산
- `03_masked_attention.py`: unmasked vs causal masked attention
- `04_multihead.py`: head별 attention map 비교
- `05_run_all.py`: 위 실습을 순서대로 실행

## 실행

```bash
python practice/04_attention_is_all_you_need/05_run_all.py
```

또는 하나씩 실행해도 된다.

## 생성 결과

```text
outputs/04_attention_is_all_you_need/
├── 01_qkv_projection.png
├── 02_scaled_dot_product_attention.png
├── 03_masked_vs_unmasked.png
└── 04_multihead_attention.png
```

## 이 실습에서 하지 않는 것

- WMT 번역 dataset 전처리
- tokenizer 학습
- encoder/decoder 전체 stack 학습
- BLEU 재현
- 논문의 모든 ablation 재현

이 폴더에서 그림을 보고 아래 질문에 답할 수 있으면 충분하다.

> 하나의 token이 다른 token을 어떤 weight로 참고하고, mask와 multi-head가 그 attention을 어떻게 바꾸는가?
