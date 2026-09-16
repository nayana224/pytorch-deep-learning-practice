#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/checkpoints/sam"
FILE="$DIR/sam_vit_b_01ec64.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"

mkdir -p "$DIR"

if [[ -s "$FILE" ]]; then
  echo "[skip] SAM ViT-B checkpoint already exists: $FILE"
  exit 0
fi

if command -v wget >/dev/null 2>&1; then
  wget -c -O "$FILE" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L -C - "$URL" -o "$FILE"
else
  echo "Error: wget or curl is required." >&2
  exit 1
fi

echo "[done] $FILE"
