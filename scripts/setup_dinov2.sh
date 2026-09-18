#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$ROOT/external/dinov2"

mkdir -p "$ROOT/external"

if [ ! -d "$DST/.git" ]; then
    git clone --depth 1 https://github.com/facebookresearch/dinov2.git "$DST"
else
    echo "[skip] official DINOv2 repo already exists: $DST"
fi

echo "[prepare] downloading/caching official dinov2_vits14 pretrained weights"

python - "$DST" <<'PY'
from pathlib import Path
import sys
import torch

repo = Path(sys.argv[1]).resolve()
model = torch.hub.load(
    str(repo),
    "dinov2_vits14",
    source="local",
    pretrained=True,
)
model.eval()

print("[ready] DINOv2 repo:", repo)
print("[ready] model:", model.__class__.__name__)
print("[ready] torch hub cache:", torch.hub.get_dir())
PY

echo
echo "DINOv2 setup complete."
echo "Practice scripts can now use the local official repo and cached weights."
