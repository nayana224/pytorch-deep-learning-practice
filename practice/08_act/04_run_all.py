"""ACT의 첫 바퀴용 핵심 메커니즘 실습을 한 번에 실행한다."""

from pathlib import Path
import runpy


THIS_DIR = Path(__file__).resolve().parent

# 실제 ALOHA 학습은 Level 3에서 따로 진행한다.
# 여기서는 action chunk, temporal ensemble, CVAE latent만 빠르게 확인한다.
scripts = [
    "01_action_chunking.py",
    "02_temporal_ensemble.py",
    "03_cvae_latent.py",
]

for script in scripts:
    path = THIS_DIR / script
    print(f"\n===== {script} 실행 =====")
    runpy.run_path(str(path), run_name="__main__")
