from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torchvision.datasets import VOCSegmentation
from deeplabv3plus import DeepLabV3Plus
from importlib.machinery import SourceFileLoader
D=SourceFileLoader("voc_data","practice/03_deeplabv3plus/01_data.py").load_module()

OUT=Path("outputs/03_deeplabv3plus")
ckpt=OUT/"model.pt"
if not ckpt.exists(): raise SystemExit("train first: python practice/03_deeplabv3plus/04_train.py")
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=DeepLabV3Plus(21,16).to(dev); model.load_state_dict(torch.load(ckpt,map_location=dev)); model.eval()
ds=VOCSegmentation("data/03_deeplabv3plus",year="2012",image_set="val",download=True)
image,mask=ds[0]; x,y=D.pair_to_tensor(image,mask,False)
with torch.no_grad(): logits,feats=model(x.unsqueeze(0).to(dev),return_features=True); pred=logits.argmax(1)[0].cpu()
error=(pred!=y)&(y!=255)
fig,axes=plt.subplots(1,4,figsize=(16,4))
axes[0].imshow(image); axes[0].set_title("input")
axes[1].imshow(y,cmap="tab20"); axes[1].set_title("GT")
axes[2].imshow(pred,cmap="tab20"); axes[2].set_title("prediction")
axes[3].imshow(error,cmap="gray"); axes[3].set_title("error map")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"05_prediction.png",dpi=150); plt.show()
print({k:tuple(v.shape) for k,v in feats.items()})
