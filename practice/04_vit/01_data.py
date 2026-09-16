from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR100
from torchvision import transforms

DATA=Path("data/04_vit"); OUT=Path("outputs/04_vit"); OUT.mkdir(parents=True,exist_ok=True)
raw=CIFAR100(DATA,train=True,download=True)
image,label=raw[0]
print("CIFAR-100 train:",len(raw),"raw size:",image.size,"label:",label,raw.classes[label])
fig,axes=plt.subplots(1,2,figsize=(8,4)); axes[0].imshow(image); axes[0].set_title("original 32x32")
resized=transforms.Resize((384,384))(image); axes[1].imshow(resized); axes[1].set_title("paper fine-tune resolution 384")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"01_cifar100.png",dpi=150); plt.show()
