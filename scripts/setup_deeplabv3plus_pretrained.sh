#!/usr/bin/env bash
set -euo pipefail

EXTERNAL_DIR="external/DeepLabV3Plus-Pytorch"
CHECKPOINT_DIR="checkpoints/deeplabv3plus"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/best_deeplabv3plus_resnet101_voc_os16.pth"
CHECKPOINT_URL="https://www.dropbox.com/s/bm3hxe7wmakaqc5/best_deeplabv3plus_resnet101_voc_os16.pth?dl=1"

mkdir -p external
mkdir -p "${CHECKPOINT_DIR}"

if [ -d "${EXTERNAL_DIR}/.git" ]; then
    echo "[skip] external DeepLabV3Plus-Pytorch already exists"
else
    git clone --depth 1         https://github.com/VainF/DeepLabV3Plus-Pytorch.git         "${EXTERNAL_DIR}"
fi

if [ -f "${CHECKPOINT_PATH}" ]; then
    echo "[skip] pretrained checkpoint already exists: ${CHECKPOINT_PATH}"
else
    echo "[download] DeepLabV3Plus-ResNet101 VOC OS16 checkpoint"
    wget -c         -O "${CHECKPOINT_PATH}"         "${CHECKPOINT_URL}"
fi

echo
echo "Ready:"
echo "  repo       : ${EXTERNAL_DIR}"
echo "  checkpoint : ${CHECKPOINT_PATH}"
