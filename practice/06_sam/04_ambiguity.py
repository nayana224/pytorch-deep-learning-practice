import matplotlib.pyplot as plt
import numpy as np
from sam_utils import load_sample,load_predictor,foreground_point,iou,OUT

image,gt,ann,_=load_sample(); predictor,_=load_predictor(); predictor.set_image(image); point=foreground_point(gt)
masks,scores,_=predictor.predict(point_coords=point,point_labels=np.array([1]),multimask_output=True)
fig,axes=plt.subplots(1,len(masks),figsize=(5*len(masks),5))
for i,(mask,score) in enumerate(zip(masks,scores)):
    axes[i].imshow(image); axes[i].imshow(mask,cmap="viridis",alpha=.5); axes[i].scatter(point[:,0],point[:,1],c="red",s=60); axes[i].set_title(f"mask {i}: predIoU={score:.3f}\nGT IoU={iou(mask,gt):.3f}"); axes[i].axis("off")
plt.tight_layout(); plt.savefig(OUT/"04_ambiguity.png",dpi=150); plt.show()
