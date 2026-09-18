"""SAM의 핵심 promptable segmentation을 한 번의 model/image embedding으로 확인한다.

point prompt, box prompt, ambiguous point의 multimask 출력을 한 프로세스에서 실행한다.
SAM model과 image embedding을 반복해서 계산하지 않는 것이 목적이다.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from sam_utils import OUT, bbox_xyxy, foreground_point, iou, load_predictor, load_sample


# 1) 같은 SA-1B sample을 모든 prompt 실험에 공통으로 사용한다.
image, gt_mask, annotation, image_path = load_sample()

# 2) 공식 pretrained SAM을 한 번만 load하고 image embedding도 한 번만 계산한다.
predictor, device = load_predictor()
predictor.set_image(image)

point = foreground_point(gt_mask)
point_label = np.array([1])
box = bbox_xyxy(annotation)

print("image:", image_path)
print("device:", device)

# ---------------------------------------------------------------------
# A. Positive point prompt
# ---------------------------------------------------------------------
point_masks, point_scores, _ = predictor.predict(
    point_coords=point,
    point_labels=point_label,
    multimask_output=False,
)

point_mask = point_masks[0]
point_score = float(point_scores[0])
point_actual_iou = iou(point_mask, gt_mask)

print("point prompt:", point.tolist())
print("point predicted IoU:", point_score)
print("point actual GT IoU:", point_actual_iou)

fig1, axes1 = plt.subplots(1, 3, figsize=(15, 5))

axes1[0].imshow(image)
axes1[0].scatter(point[:, 0], point[:, 1], s=80)
axes1[0].set_title("Image + positive point")

axes1[1].imshow(gt_mask, cmap="gray")
axes1[1].set_title("SA-1B GT")

axes1[2].imshow(image)
axes1[2].imshow(point_mask, alpha=0.5)
axes1[2].set_title(
    f"Point prediction\nPred IoU={point_score:.3f}, GT IoU={point_actual_iou:.3f}"
)

for ax in axes1:
    ax.axis("off")

fig1.tight_layout()
fig1.savefig(OUT / "02_point.png", dpi=150)

# ---------------------------------------------------------------------
# B. Box prompt
# ---------------------------------------------------------------------
box_masks, box_scores, _ = predictor.predict(
    box=box,
    multimask_output=False,
)

box_mask = box_masks[0]
box_score = float(box_scores[0])
box_actual_iou = iou(box_mask, gt_mask)

print("box prompt:", box.tolist())
print("box predicted IoU:", box_score)
print("box actual GT IoU:", box_actual_iou)

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))

axes2[0].imshow(image)
axes2[0].add_patch(
    Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        fill=False,
        linewidth=2,
    )
)
axes2[0].set_title("Image + box prompt")

axes2[1].imshow(image)
axes2[1].imshow(box_mask, alpha=0.5)
axes2[1].set_title(
    f"Box prediction\nPred IoU={box_score:.3f}, GT IoU={box_actual_iou:.3f}"
)

for ax in axes2:
    ax.axis("off")

fig2.tight_layout()
fig2.savefig(OUT / "03_box.png", dpi=150)

# ---------------------------------------------------------------------
# C. Ambiguous point -> multiple masks
# ---------------------------------------------------------------------
multi_masks, multi_scores, _ = predictor.predict(
    point_coords=point,
    point_labels=point_label,
    multimask_output=True,
)

fig3, axes3 = plt.subplots(
    1,
    len(multi_masks),
    figsize=(5 * len(multi_masks), 5),
)

for index, (mask, score) in enumerate(zip(multi_masks, multi_scores)):
    actual_iou = iou(mask, gt_mask)

    axes3[index].imshow(image)
    axes3[index].imshow(mask, alpha=0.5)
    axes3[index].scatter(point[:, 0], point[:, 1], s=60)
    axes3[index].set_title(
        f"Mask {index}\nPred IoU={score:.3f}\nGT IoU={actual_iou:.3f}"
    )
    axes3[index].axis("off")

fig3.tight_layout()
fig3.savefig(OUT / "04_ambiguity.png", dpi=150)

plt.show()
