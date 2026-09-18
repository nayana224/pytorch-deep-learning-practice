"""Attention Is All You Need의 최소 실습을 순서대로 실행한다."""

from pathlib import Path
import runpy


THIS_DIR = Path(__file__).resolve().parent

scripts = [
    "01_qkv.py",
    "02_scaled_dot_product.py",
    "03_masked_attention.py",
    "04_multihead.py",
]

for script in scripts:
    path = THIS_DIR / script
    print(f"\n===== {script} 실행 =====")
    runpy.run_path(str(path), run_name="__main__")
