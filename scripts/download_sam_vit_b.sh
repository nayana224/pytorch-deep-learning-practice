#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/checkpoints/sam"
mkdir -p "$DIR"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
FILE="$DIR/sam_vit_b_01ec64.pth"
if command -v wget >/dev/null 2>&1; then wget -O "$FILE" "$URL"; else curl -L "$URL" -o "$FILE"; fi
echo "$FILE"
