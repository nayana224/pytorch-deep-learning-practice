import torch
from torchvision.datasets import OxfordIIITPet
from common import load_model,transform,extract

model,device=load_model(); ds=OxfordIIITPet("data/05_dinov2",split="test",download=True,transform=transform())
x,y=ds[0]; x=x.unsqueeze(0).to(device)
with torch.no_grad(): cls,patch=extract(model,x)
print("input:",x.shape,"CLS:",cls.shape,"patch tokens:",patch.shape)
side=int(patch.shape[1]**0.5); print("patch grid:",side,"x",side,"patch size implied:",224//side)
print("CLS norm:",cls.norm(dim=-1).item(),"patch mean norm:",patch.norm(dim=-1).mean().item())
