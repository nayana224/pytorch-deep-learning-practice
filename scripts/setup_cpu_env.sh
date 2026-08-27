#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pytorch-dl-practice-cpu"
ENV_FILE="environment.cpu.yml"

if ! command -v conda >/dev/null 2>&1; then
    echo "[ERROR] conda 명령을 찾을 수 없습니다. Miniconda를 먼저 설치하세요."
    exit 1
fi

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "[ERROR] ${ENV_FILE} 파일이 없습니다. 저장소 루트에서 실행하세요."
    exit 1
fi

# 현재 shell에서 conda 명령을 일관되게 사용할 수 있게 한다.
eval "$(conda shell.bash hook)"

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "[INFO] 기존 CPU Conda 환경을 갱신합니다: ${ENV_NAME}"
    conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
    echo "[INFO] CPU Conda 환경을 생성합니다: ${ENV_NAME}"
    conda env create --file "${ENV_FILE}"
fi

echo
conda activate "${ENV_NAME}"

echo "[INFO] CPU 환경 설정 완료"
python --version
python -c 'import torch; print("torch:", torch.__version__); print("torch CUDA runtime:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available()); x = torch.rand(3, 3); print("tensor device:", x.device)'

echo
echo "다음 명령으로 환경을 활성화하세요:"
echo "  conda activate ${ENV_NAME}"
echo "그 다음 환경 검증:"
echo "  python scripts/00_check_environment.py"
