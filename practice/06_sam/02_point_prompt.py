import matplotlib.pyplot as plt
import numpy as np
from sam_utils import load_sample,load_predictor,foreground_point,iou,OUT

image,gt,ann,_=load_sample(); predictor,_=load_predictor(); predictor.set_image(image); point=foreground_point(gt)
masks,scores,_=predictor.predict(point_coords=point,point_labels=np.array([1]),multimask_output=False); pred=masks[0]
print("point:",point.tolist(),"predicted IoU score:",float(scores[0]),"GT IoU:",iou(pred,gt))
fig,axes=plt.subplots(1,3,figsize=(15,5)); axes[0].imshow(image); axes[0].scatter(point[:,0],point[:,1],c="lime",s=80); axes[0].set_title("foreground point")
axes[1].imshow(gt,cmap="gray"); axes[1].set_title("SA-1B GT")
axes[2].imshow(image); axes[2].imshow(pred,cmap="Blues",alpha=.5); axes[2].set_title("SAM point prediction")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"02_point.png",dpi=150); plt.show()
