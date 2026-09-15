#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/02_unet/isbi2012"
ARCHIVE="$DATA_DIR/ISBI-2012-challenge.zip"
URL="https://downloads.imagej.net/ISBI-2012-challenge.zip"

mkdir -p "$DATA_DIR"

echo "[1/3] Downloading ISBI 2012 challenge archive"
if command -v wget >/dev/null 2>&1; then
  wget -O "$ARCHIVE" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$ARCHIVE"
else
  echo "Error: wget or curl is required." >&2
  exit 1
fi

echo "[2/3] Extracting archive"
unzip -o "$ARCHIVE" -d "$DATA_DIR"

echo "[3/3] Files"
find "$DATA_DIR" -maxdepth 2 -type f | sort

echo
echo "Dataset prepared at: $DATA_DIR"
echo "Next: inspect the actual TIFF filenames, then open practice/02_unet/01_data.py"
