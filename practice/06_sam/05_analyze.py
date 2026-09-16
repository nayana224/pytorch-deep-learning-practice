import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from sam_utils import load_sample,CKPT,OUT
import torch

image,gt,ann,_=load_sample(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); sam=sam_model_registry["vit_b"](checkpoint=str(CKPT)).to(dev)
generator=SamAutomaticMaskGenerator(sam,points_per_side=16); masks=generator.generate(image); masks=sorted(masks,key=lambda m:m["area"],reverse=True)
print("generated masks:",len(masks),"largest areas:",[m["area"] for m in masks[:10]])
fig,ax=plt.subplots(figsize=(8,6)); ax.imshow(image)
for m in masks[:20]: ax.contour(m["segmentation"],levels=[.5],linewidths=.7)
ax.set_title("automatic mask generation: top-20 by area"); ax.axis("off"); plt.tight_layout(); plt.savefig(OUT/"05_auto_masks.png",dpi=150); plt.show()
