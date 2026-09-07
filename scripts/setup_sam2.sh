#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pytorch-dl-notebook"
SAM2_DIR="external/sam2"

if ! command -v conda >/dev/null 2>&1; then
  echo "[오류] conda를 찾을 수 없습니다. 먼저 scripts/setup_notebook_env.sh를 실행하세요."
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

mkdir -p external

if [ -d "$SAM2_DIR/.git" ]; then
  echo "기존 SAM2 저장소를 업데이트합니다."
  git -C "$SAM2_DIR" pull --ff-only
else
  echo "Meta 공식 SAM2 저장소를 clone합니다."
  git clone https://github.com/facebookresearch/sam2.git "$SAM2_DIR"
fi

python -m pip install --upgrade pip
python -m pip install -e "$SAM2_DIR[notebooks]"

echo
echo "SAM2 설치 완료"
echo "공식 checkpoint는 external/sam2/checkpoints/ 아래에 두는 것을 권장합니다."
echo "처음 실습은 sam2.1_hiera_tiny checkpoint로 시작하는 것을 권장합니다."
echo "자세한 checkpoint 다운로드 방법은 README를 확인하세요."
