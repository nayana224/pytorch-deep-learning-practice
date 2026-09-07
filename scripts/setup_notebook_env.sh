#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pytorch-dl-notebook"
ENV_FILE="environment.notebook.yml"

if ! command -v conda >/dev/null 2>&1; then
  echo "[오류] conda 명령을 찾을 수 없습니다. Miniconda/Anaconda를 먼저 설치하세요."
  exit 1
fi

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[$ENV_NAME] 환경이 이미 있습니다. environment.notebook.yml 기준으로 업데이트합니다."
  conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
else
  echo "[$ENV_NAME] 환경을 새로 만듭니다."
  conda env create -f "$ENV_FILE"
fi

echo
echo "환경 구성이 끝났습니다."
echo "다음 명령을 실행하세요:"
echo "  conda activate $ENV_NAME"
echo "  python -m ipykernel install --user --name $ENV_NAME --display-name 'Python ($ENV_NAME)'"
echo "  python scripts/00_check_environment.py"
echo
echo "VSCode에서는 .ipynb를 연 뒤 오른쪽 위 Kernel에서 'Python ($ENV_NAME)'을 선택하세요."
