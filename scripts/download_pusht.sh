#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/data/07_diffusion_policy"
ZIP="$DIR/pusht.zip"
URL="https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip"
mkdir -p "$DIR"
if command -v wget >/dev/null 2>&1; then
  wget -O "$ZIP" "$URL"
else
  curl -L "$URL" -o "$ZIP"
fi
unzip -o "$ZIP" -d "$DIR"
echo "Push-T data candidates:"
find "$DIR" -maxdepth 4 \( -name '*.zarr' -o -name '*.zip' \) -print
