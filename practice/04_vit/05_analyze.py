from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import CIFAR100
from torchvision import transforms
from vit import vit_b16,vit_tiny16

OUT=Path("outputs/04_vit"); ckpt=torch.load(OUT/"tiny.pt",map_location="cpu"); size=ckpt["size"]
model=vit_tiny16(100,size) if ckpt["variant"]=="tiny" else vit_b16(100,size); model.load_state_dict(ckpt["model"]); model.eval()
ds=CIFAR100("data/04_vit",train=False,download=True); image,label=ds[0]
tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])]); x=tf(image).unsqueeze(0)
with torch.no_grad(): logits,f=model(x,return_attention=True)
att=f["attentions"][-1][0].mean(0)[0,1:]; grid=size//16; att=att.reshape(grid,grid)
att=torch.nn.functional.interpolate(att[None,None],size=(size,size),mode="bilinear",align_corners=False)[0,0]
fig,axes=plt.subplots(1,2,figsize=(9,4)); axes[0].imshow(image.resize((size,size))); axes[0].set_title(f"GT={ds.classes[label]} pred={ds.classes[logits.argmax(1).item()]}")
axes[1].imshow(image.resize((size,size))); axes[1].imshow(att,cmap="magma",alpha=.55); axes[1].set_title("last-layer CLS attention")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"05_attention.png",dpi=150); plt.show()
