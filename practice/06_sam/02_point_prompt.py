"""SAM에서 point prompt 하나가 mask prediction을 어떻게 바꾸는지 확인한다.

전체 SAM 학습을 재현하지 않는다.
이미 계산된 image embedding에 positive point prompt를 주고,
predicted mask와 predicted-IoU가 실제 GT와 어떤 관계인지 본다.
"""

import matplotlib.pyplot as plt
import numpy as np

from sam_utils import OUT, foreground_point, iou, load_predictor, load_sample


# 1) SA-1B sample과 GT mask를 읽는다.
image, gt_mask, annotation, image_path = load_sample()

# 2) 공식 pretrained SAM predictor를 준비하고,
#    한 번만 image embedding을 계산한다.
predictor, device = load_predictor()
predictor.set_image(image)

# 3) GT 내부의 한 점을 positive point prompt로 사용한다.
#    실제 사용 환경에서는 사용자가 클릭한 점에 해당한다.
point = foreground_point(gt_mask)
point_label = np.array([1])

# 4) 같은 image embedding + point prompt → mask
#    multimask_output=False이므로 현재는 단일 mask만 받는다.
masks, scores, logits = predictor.predict(
    point_coords=point,
    point_labels=point_label,
    multimask_output=False,
)

predicted_mask = masks[0]
predicted_iou_score = float(scores[0])
actual_iou = iou(predicted_mask, gt_mask)

print("image:", image_path)
print("device:", device)
print("point:", point.tolist())
print("SAM predicted IoU:", predicted_iou_score)
print("actual GT IoU:", actual_iou)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].scatter(point[:, 0], point[:, 1], s=80)
axes[0].set_title("image + positive point")

axes[1].imshow(gt_mask, cmap="gray")
axes[1].set_title("SA-1B GT")

axes[2].imshow(image)
axes[2].imshow(predicted_mask, alpha=0.5)
axes[2].set_title("SAM predicted mask")

for ax in axes:
    ax.axis("off")

fig.suptitle("SAM: prompt-conditioned mask prediction from one image embedding")
fig.tight_layout()
fig.savefig(OUT / "02_point.png", dpi=150)
plt.show()
