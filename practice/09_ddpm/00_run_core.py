"""이 논문의 첫 바퀴 핵심 실습을 순서대로 실행한다.

전체 학습을 자동으로 돌리지 않고, README에 정의된 핵심 메커니즘만 확인한다.
core runner에서는 GUI 창을 띄우지 않고 그림을 outputs/<paper>/에 저장한다.
개별 .py 파일을 직접 실행하면 기존처럼 matplotlib 창을 볼 수 있다.
"""

from pathlib import Path
import os
import subprocess
import sys


PRACTICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRACTICE_DIR.parents[1]

CORE_SCRIPTS = [
    "01_forward_noising.py",
    "02_noise_target.py",
    "03_reconstruct_x0.py",
]


def main() -> None:
    print(f"[core] {PRACTICE_DIR.name}")
    print("[core] 핵심 메커니즘 실습을 순서대로 실행합니다.\n")

    # Agg backend를 사용하면 plt.show()가 GUI 창을 열지 않는다.
    # 각 스크립트의 savefig() 결과는 그대로 outputs/에 저장된다.
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PAPER_CORE_RUN"] = "1"

    for script_name in CORE_SCRIPTS:
        script_path = PRACTICE_DIR / script_name

        print("=" * 72)
        print(f"[run] {script_path.relative_to(REPO_ROOT)}")
        print("=" * 72)

        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )
        print()

    print("[done] outputs/ 아래 생성된 그림을 확인하세요.")


if __name__ == "__main__":
    main()
