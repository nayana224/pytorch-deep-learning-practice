"""이 논문의 첫 바퀴 핵심 실습을 순서대로 실행한다.

전체 논문 구현을 다시 돌리는 스크립트가 아니다.
README에 정의된 핵심 메커니즘만 빠르게 확인하고,
생성된 그림은 outputs/<paper>/에서 바로 확인한다.
"""

from pathlib import Path
import subprocess
import sys


PRACTICE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PRACTICE_DIR.parents[1]

CORE_SCRIPTS = [
    "02_features.py",
    "03_pca.py",
    "05_analyze.py",
]


def main() -> None:
    print(f"[core] {PRACTICE_DIR.name}")
    print("[core] 핵심 메커니즘 실습을 순서대로 실행합니다.\n")

    for script_name in CORE_SCRIPTS:
        script_path = PRACTICE_DIR / script_name

        print("=" * 72)
        print(f"[run] {script_path.relative_to(REPO_ROOT)}")
        print("=" * 72)

        # 각 파일을 별도 Python process로 실행한다.
        # 한 스크립트의 전역 변수나 matplotlib 상태가 다음 실습에
        # 영향을 주지 않게 하기 위한 의도적인 분리다.
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=REPO_ROOT,
            check=True,
        )

        print()

    print("[done] outputs/ 아래 생성된 그림을 확인하세요.")


if __name__ == "__main__":
    main()
