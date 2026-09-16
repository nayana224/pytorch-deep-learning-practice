#!/usr/bin/env bash
set -euo pipefail
python -m pip install 'git+https://github.com/facebookresearch/segment-anything.git'
python -m pip install opencv-python pycocotools matplotlib
