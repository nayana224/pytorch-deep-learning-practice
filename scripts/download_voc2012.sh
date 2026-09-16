#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/data/03_deeplabv3plus"
ARCHIVE="$DIR/VOCtrainval_11-May-2012.tar"
URL="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
READY="$DIR/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"

mkdir -p "$DIR"

if [[ -f "$READY" ]]; then
  echo "[skip] PASCAL VOC 2012 already exists: $DIR/VOCdevkit/VOC2012"
  exit 0
fi

echo "[download/resume] PASCAL VOC 2012"
if command -v wget >/dev/null 2>&1; then
  wget -c -O "$ARCHIVE" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L -C - "$URL" -o "$ARCHIVE"
else
  echo "Error: wget or curl is required." >&2
  exit 1
fi

echo "[extract] $ARCHIVE"
tar -xf "$ARCHIVE" -C "$DIR"

if [[ ! -f "$READY" ]]; then
  echo "Error: VOC 2012 extraction did not produce the expected files." >&2
  exit 1
fi

echo "[done] $DIR/VOCdevkit/VOC2012"
