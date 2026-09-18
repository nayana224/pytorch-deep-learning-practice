"""SAM 논문 실습 코드.

같은 image embedding에 point/box prompt를 주었을 때
mask와 multimask ambiguity가 어떻게 달라지는지 확인하기 위한 코드다.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sam_utils import OUT, bbox_xyxy, iou, load_predictor, load_sample


# 1. SA-1B image + one GT mask
image, gt_mask, annotation, image_path = load_sample()

# 2. Convert the GT annotation box to xyxy prompt coordinates
box = bbox_xyxy(annotation)

# 3. Official SAM predictor
predictor, device = load_predictor()
predictor.set_image(image)

# 4. image + box prompt -> predicted mask
masks, scores, logits = predictor.predict(
    box=box,
    multimask_output=False,
)

predicted_mask = masks[0]
predicted_iou_score = float(scores[0])
actual_iou = iou(predicted_mask, gt_mask)

print("image:", image_path)
print("box:", box.tolist())
print("predicted IoU score:", predicted_iou_score)
print("actual GT IoU:", actual_iou)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(image)
axes[0].add_patch(
    Rectangle(
        (box[0], box[1]),
        box[2] - box[0],
        box[3] - box[1],
        fill=False,
        linewidth=2,
    )
)
axes[0].set_title("image + box prompt")

axes[1].imshow(image)
axes[1].imshow(predicted_mask, alpha=0.5)
axes[1].set_title("SAM box prediction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "03_box.png", dpi=150)
plt.show()
