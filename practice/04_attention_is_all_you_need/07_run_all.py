"""Attention Is All You Need의 Level 2 핵심 실습을 한 번에 실행한다."""

from pathlib import Path
import runpy


THIS_DIR = Path(__file__).resolve().parent

# 논문 전체 번역 모델을 학습하지 않고,
# self-attention을 이해하는 데 필요한 메커니즘만 순서대로 확인한다.
scripts = [
    "01_qkv.py",
    "02_scaled_dot_product.py",
    "03_masked_attention.py",
    "04_multihead.py",
    "05_positional_encoding.py",
    "06_cross_attention.py",
]

for script in scripts:
    path = THIS_DIR / script
    print(f"\n===== {script} 실행 =====")
    runpy.run_path(str(path), run_name="__main__")
