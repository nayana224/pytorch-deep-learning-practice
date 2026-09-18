"""SAM 논문 실습 코드.

같은 image embedding에 point/box prompt를 주었을 때
mask와 multimask ambiguity가 어떻게 달라지는지 확인하기 위한 코드다.
"""

import matplotlib.pyplot as plt
from sam_utils import load_sample,OUT

image,gt,ann,path=load_sample()
print("SA-1B image:",path)
print("image:",image.shape,image.dtype,"GT:",gt.shape,gt.dtype,"mask area:",gt.sum(),"bbox:",ann["bbox"])
fig,axes=plt.subplots(1,2,figsize=(12,5)); axes[0].imshow(image); axes[0].set_title("SA-1B image")
axes[1].imshow(image); axes[1].imshow(gt,cmap="Reds",alpha=.45); axes[1].set_title("released SA-1B mask")
for ax in axes: ax.axis("off")
plt.tight_layout(); plt.savefig(OUT/"01_sa1b.png",dpi=150); plt.show()
