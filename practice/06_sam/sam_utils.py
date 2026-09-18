"""SAM 논문 실습 코드.

같은 image embedding에 point/box prompt를 주었을 때
mask와 multimask ambiguity가 어떻게 달라지는지 확인하기 위한 코드다.
"""

from pathlib import Path
import json

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils
from segment_anything import SamPredictor, sam_model_registry


DATA_DIR = Path("data/06_sam/sa1b")
CHECKPOINT = Path("checkpoints/sam/sam_vit_b_01ec64.pth")
OUTPUT_DIR = Path("outputs/06_sam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing script compatibility
OUT = OUTPUT_DIR


def find_sample():
    image_paths = sorted(
        list(DATA_DIR.rglob("*.jpg"))
        + list(DATA_DIR.rglob("*.jpeg"))
        + list(DATA_DIR.rglob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(
            "Put an official SA-1B subset under data/06_sam/sa1b."
        )

    for image_path in image_paths:
        annotation_path = image_path.with_suffix(".json")
        if annotation_path.exists():
            return image_path, annotation_path

    raise FileNotFoundError("No matching SA-1B json annotation found.")


def load_sample():
    image_path, annotation_path = find_sample()

    image_bgr = cv2.imread(str(image_path))
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    metadata = json.loads(annotation_path.read_text())
    annotation = metadata["annotations"][0]

    rle = annotation["segmentation"]
    if isinstance(rle.get("counts"), str):
        rle["counts"] = rle["counts"].encode("utf-8")

    gt_mask = mask_utils.decode(rle).astype(bool)
    return image, gt_mask, annotation, image_path


def load_predictor():
    if not CHECKPOINT.exists():
        raise FileNotFoundError("Run bash scripts/download_sam_vit_b.sh")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sam = sam_model_registry["vit_b"](checkpoint=str(CHECKPOINT))
    sam = sam.to(device)

    predictor = SamPredictor(sam)
    return predictor, device


def foreground_point(mask):
    y_coordinates, x_coordinates = np.where(mask)
    middle = len(x_coordinates) // 2

    return np.array(
        [[x_coordinates[middle], y_coordinates[middle]]],
        dtype=np.float32,
    )


def bbox_xyxy(annotation):
    x, y, width, height = annotation["bbox"]
    return np.array([x, y, x + width, y + height], dtype=np.float32)


def iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 1.0
    return float(intersection / union)
