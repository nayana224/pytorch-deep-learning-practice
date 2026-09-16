from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import OxfordIIITPet
from common import load_model,transform,extract

OUT=Path("outputs/05_dinov2"); OUT.mkdir(parents=True,exist_ok=True)
model,device=load_model(); raw=OxfordIIITPet("data/05_dinov2",split="test",download=True); tf=transform()
images=[]; tokens=[]
for i in [0,1,2]:
    img,_=raw[i]; images.append(img); x=tf(img).unsqueeze(0).to(device)
    with torch.no_grad(): _,p=extract(model,x)
    tokens.append(p[0].cpu())
all_tokens=torch.cat(tokens,0); centered=all_tokens-all_tokens.mean(0,keepdim=True); _,_,v=torch.pca_lowrank(centered,q=3); rgb=centered@v; rgb=(rgb-rgb.amin(0))/(rgb.amax(0)-rgb.amin(0)+1e-6)
fig,axes=plt.subplots(2,3,figsize=(10,7)); offset=0
for col,(img,p) in enumerate(zip(images,tokens)):
    n=p.shape[0]; side=int(n**.5); axes[0,col].imshow(img); axes[0,col].axis("off"); axes[0,col].set_title("input")
    axes[1,col].imshow(rgb[offset:offset+n].reshape(side,side,3)); axes[1,col].axis("off"); axes[1,col].set_title("patch PCA RGB"); offset+=n
plt.tight_layout(); plt.savefig(OUT/"03_patch_pca.png",dpi=150); plt.show()
