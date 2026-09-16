from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet
from common import load_model,transform,extract

OUT=Path("outputs/05_dinov2"); OUT.mkdir(parents=True,exist_ok=True); model,device=load_model(); ds=OxfordIIITPet("data/05_dinov2",split="test",download=True); tf=transform()
ids=list(range(min(100,len(ds)))); feats=[]
for i in ids:
    x=tf(ds[i][0]).unsqueeze(0).to(device)
    with torch.no_grad(): feats.append(F.normalize(extract(model,x)[0],dim=-1).cpu())
feats=torch.cat(feats); q=0; sim=feats@feats[q]; nn_ids=sim.topk(6).indices.tolist()
fig,axes=plt.subplots(1,6,figsize=(15,3))
for ax,i in zip(axes,nn_ids): ax.imshow(ds[i][0]); ax.set_title(f"{ds.classes[ds[i][1]]}\ncos={sim[i]:.2f}"); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"05_nearest_neighbors.png",dpi=150); plt.show()
print("Query + nearest-neighbor retrieval on frozen CLS features. Inspect semantic matches and failures.")
