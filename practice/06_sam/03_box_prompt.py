import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sam_utils import load_sample,load_predictor,bbox_xyxy,iou,OUT

image,gt,ann,_=load_sample(); predictor,_=load_predictor(); predictor.set_image(image); box=bbox_xyxy(ann)
masks,scores,_=predictor.predict(box=box,multimask_output=False); pred=masks[0]
print("box:",box.tolist(),"score:",float(scores[0]),"GT IoU:",iou(pred,gt))
fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].imshow(image); axes[0].add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor="lime",linewidth=2)); axes[0].set_title("GT-derived box prompt")
axes[1].imshow(image); axes[1].imshow(pred,cmap="Blues",alpha=.5); axes[1].set_title("SAM box prediction")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"03_box.png",dpi=150); plt.show()
