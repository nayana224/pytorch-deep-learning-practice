#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/data/10_diffusion_policy"
ZIP="$DIR/pusht.zip"
URL="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"

mkdir -p "$DIR"

if find "$DIR" -type d -name '*.zarr' -print -quit | grep -q .; then
  echo "[skip] Push-T dataset already exists under: $DIR"
  exit 0
fi

echo "[download/resume] Push-T"
if command -v wget >/dev/null 2>&1; then
  wget -c -O "$ZIP" "$URL"
elif command -v curl >/dev/null 2>&1; then
  curl -L -C - "$URL" -o "$ZIP"
else
  echo "오류: wget 또는 curl이 필요합니다." >&2
  exit 1
fi

unzip -o "$ZIP" -d "$DIR"

echo "[done] Push-T 데이터 후보:"
find "$DIR" -maxdepth 4 -name '*.zarr' -print
