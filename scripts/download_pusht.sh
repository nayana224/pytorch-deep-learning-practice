#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/data/07_diffusion_policy"
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
  echo "Error: wget or curl is required." >&2
  exit 1
fi

unzip -o "$ZIP" -d "$DIR"

echo "[done] Push-T data candidates:"
find "$DIR" -maxdepth 4 -name '*.zarr' -print
