import torch
from vit import vit_b16, vit_tiny16

for name,model,size in [("ViT-B/16",vit_b16(100,384),384),("scaled-tiny/16",vit_tiny16(100,224),224)]:
    x=torch.randn(1,3,size,size)
    model.eval()
    with torch.no_grad(): logits,f=model(x,return_features=True,return_attention=True)
    print("\n",name)
    print("input:",x.shape,"patch embeddings:",f["patches"].shape,"tokens(+CLS):",f["tokens"].shape,"logits:",logits.shape)
    print("last attention:",f["attentions"][-1].shape)
    print("params:",sum(p.numel() for p in model.parameters()))
