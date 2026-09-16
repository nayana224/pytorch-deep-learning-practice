from pathlib import Path
import json
import numpy as np
import cv2
from pycocotools import mask as mask_utils
import torch
from segment_anything import SamPredictor, sam_model_registry

DATA=Path("data/06_sam/sa1b")
CKPT=Path("checkpoints/sam/sam_vit_b_01ec64.pth")
OUT=Path("outputs/06_sam"); OUT.mkdir(parents=True,exist_ok=True)


def find_sample():
    images=sorted(list(DATA.rglob("*.jpg"))+list(DATA.rglob("*.jpeg"))+list(DATA.rglob("*.png")))
    if not images:
        raise FileNotFoundError("Put an official SA-1B shard/subset under data/06_sam/sa1b after accepting the SA-1B license.")
    for image_path in images:
        json_path=image_path.with_suffix(".json")
        if json_path.exists(): return image_path,json_path
    raise FileNotFoundError("Could not find matching SA-1B .json annotation next to an image")


def load_sample():
    image_path,json_path=find_sample()
    image=cv2.cvtColor(cv2.imread(str(image_path)),cv2.COLOR_BGR2RGB)
    meta=json.loads(json_path.read_text())
    annotations=meta.get("annotations",[])
    if not annotations: raise ValueError("SA-1B json has no annotations")
    ann=annotations[0]
    rle=ann["segmentation"]
    if isinstance(rle.get("counts"),str): rle["counts"]=rle["counts"].encode("utf-8")
    gt=mask_utils.decode(rle).astype(bool)
    return image,gt,ann,image_path


def load_predictor():
    if not CKPT.exists(): raise FileNotFoundError("Run bash scripts/download_sam_vit_b.sh")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam=sam_model_registry["vit_b"](checkpoint=str(CKPT)).to(device)
    return SamPredictor(sam),device


def foreground_point(mask):
    ys,xs=np.where(mask)
    i=len(xs)//2
    return np.array([[xs[i],ys[i]]],dtype=np.float32)


def bbox_xyxy(ann):
    x,y,w,h=ann["bbox"]
    return np.array([x,y,x+w,y+h],dtype=np.float32)


def iou(a,b):
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return float(inter/union) if union else 1.0
