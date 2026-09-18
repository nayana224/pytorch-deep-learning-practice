"""DDPM의 Level 2 핵심 메커니즘 실습을 한 번에 실행한다."""

from pathlib import Path
import runpy


THIS_DIR = Path(__file__).resolve().parent

# 전체 U-Net diffusion model을 학습하지 않고,
# forward noising -> epsilon target -> x0 복원 연결만 확인한다.
scripts = [
    "01_forward_noising.py",
    "02_noise_target.py",
    "03_reconstruct_x0.py",
]

for script in scripts:
    path = THIS_DIR / script
    print(f"\n===== {script} 실행 =====")
    runpy.run_path(str(path), run_name="__main__")
