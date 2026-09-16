import matplotlib.pyplot as plt
import numpy as np

from sam_utils import OUT, foreground_point, iou, load_predictor, load_sample


# 1. SA-1B image + GT mask
image, gt_mask, annotation, image_path = load_sample()

# 2. Official SAM predictor
predictor, device = load_predictor()
predictor.set_image(image)

# 3. One positive point prompt
point = foreground_point(gt_mask)
point_label = np.array([1])

# 4. image + prompt -> mask / predicted IoU score
masks, scores, logits = predictor.predict(
    point_coords=point,
    point_labels=point_label,
    multimask_output=False,
)

predicted_mask = masks[0]
predicted_iou_score = float(scores[0])
actual_iou = iou(predicted_mask, gt_mask)

print("image:", image_path)
print("point:", point.tolist())
print("predicted IoU score:", predicted_iou_score)
print("actual GT IoU:", actual_iou)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(image)
axes[0].scatter(point[:, 0], point[:, 1], s=80)
axes[0].set_title("image + point prompt")

axes[1].imshow(gt_mask, cmap="gray")
axes[1].set_title("SA-1B GT")

axes[2].imshow(image)
axes[2].imshow(predicted_mask, alpha=0.5)
axes[2].set_title("SAM prediction")

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig(OUT / "02_point.png", dpi=150)
plt.show()
