#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/02_unet/isbi2012"
ARCHIVE="$DATA_DIR/ISBI-2012-challenge.zip"
URL="https://downloads.imagej.net/ISBI-2012-challenge.zip"

mkdir -p "$DATA_DIR"

required=(
  "$DATA_DIR/train-volume.tif"
  "$DATA_DIR/train-labels.tif"
  "$DATA_DIR/test-volume.tif"
  "$DATA_DIR/test-labels.tif"
)

ready=true
for file in "${required[@]}"; do
  if [[ ! -f "$file" ]]; then
    ready=false
    break
  fi
done

if [[ "$ready" == true ]]; then
  echo "[skip] ISBI 2012 already exists: $DATA_DIR"
  exit 0
fi

echo "[download/resume] ISBI 2012 challenge archive"
if command -v wget >/dev/null 2>&1; then
  wget -c -O "$ARCHIVE" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L -C - "$URL" -o "$ARCHIVE"
else
  echo "Error: wget or curl is required." >&2
  exit 1
fi

echo "[extract] archive"
unzip -o "$ARCHIVE" -d "$DATA_DIR"

echo "[done] $DATA_DIR"
