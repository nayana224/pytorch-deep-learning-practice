#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DST="$ROOT/external/diffusion_policy"
if [ ! -d "$DST/.git" ]; then
  git clone https://github.com/real-stanford/diffusion_policy.git "$DST"
else
  echo "official repo already exists: $DST"
fi
cat <<'EOF'

Official Diffusion Policy source is now under external/diffusion_policy.
The original project has a non-trivial robotics environment. For full official training,
use the environment instructions from external/diffusion_policy/README.md.
The practice scripts import model/dataset code from this unmodified official checkout.
EOF
