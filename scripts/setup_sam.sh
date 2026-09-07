#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pytorch-dl-notebook"

if ! command -v conda >/dev/null 2>&1; then
  echo "[오류] conda를 찾을 수 없습니다. 먼저 scripts/setup_notebook_env.sh를 실행하세요."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install "git+https://github.com/facebookresearch/segment-anything.git"

echo
echo "SAM 패키지 설치 완료"
echo "checkpoint는 용량이 크므로 자동으로 받지 않습니다."
echo "README의 'SAM 추가 설정'에서 권장 checkpoint와 저장 위치를 확인하세요."
