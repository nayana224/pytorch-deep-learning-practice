from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms

OUT=Path("outputs/04_vit"); OUT.mkdir(parents=True,exist_ok=True)
img,_=CIFAR100("data/04_vit",train=True,download=True)[0]
x=transforms.ToTensor()(transforms.Resize((384,384))(img))
patches=x.unfold(1,16,16).unfold(2,16,16).permute(1,2,0,3,4).reshape(-1,3,16,16)
print("image:",x.shape,"patches:",patches.shape,"sequence length:",patches.shape[0])
fig,axes=plt.subplots(4,8,figsize=(10,5))
for i,ax in enumerate(axes.flat): ax.imshow(patches[i].permute(1,2,0)); ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"02_patches.png",dpi=150); plt.show()
