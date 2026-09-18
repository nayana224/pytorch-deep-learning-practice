"""SAM 논문 실습 코드.

같은 image embedding에 point/box prompt를 주었을 때
mask와 multimask ambiguity가 어떻게 달라지는지 확인하기 위한 코드다.
"""

import matplotlib.pyplot as plt
import numpy as np

from sam_utils import OUT, foreground_point, iou, load_predictor, load_sample


image, gt_mask, annotation, image_path = load_sample()

predictor, device = load_predictor()
predictor.set_image(image)

point = foreground_point(gt_mask)
point_label = np.array([1])

# The paper makes SAM ambiguity-aware by predicting multiple masks
# for a single ambiguous prompt.
masks, scores, logits = predictor.predict(
    point_coords=point,
    point_labels=point_label,
    multimask_output=True,
)

fig, axes = plt.subplots(1, len(masks), figsize=(5 * len(masks), 5))

for index, (mask, score) in enumerate(zip(masks, scores)):
    axes[index].imshow(image)
    axes[index].imshow(mask, alpha=0.5)
    axes[index].scatter(point[:, 0], point[:, 1], s=60)

    actual_iou = iou(mask, gt_mask)
    axes[index].set_title(
        f"mask {index}\n"
        f"pred IoU={score:.3f}\n"
        f"GT IoU={actual_iou:.3f}"
    )
    axes[index].axis("off")

plt.tight_layout()
plt.savefig(OUT / "04_ambiguity.png", dpi=150)
plt.show()
