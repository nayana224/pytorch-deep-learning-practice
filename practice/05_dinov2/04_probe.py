import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import OxfordIIITPet
from common import load_model,transform,extract

p=argparse.ArgumentParser(); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=64); a=p.parse_args()
model,device=load_model(); tf=transform(); train=DataLoader(OxfordIIITPet("data/05_dinov2",split="trainval",download=True,transform=tf),batch_size=a.batch_size,shuffle=True,num_workers=4); test=DataLoader(OxfordIIITPet("data/05_dinov2",split="test",download=True,transform=tf),batch_size=a.batch_size,num_workers=4)
for p0 in model.parameters(): p0.requires_grad=False
sample=next(iter(train))[0][:1].to(device)
with torch.no_grad(): dim=extract(model,sample)[0].shape[-1]
probe=nn.Linear(dim,37).to(device); opt=torch.optim.SGD(probe.parameters(),lr=.05,momentum=.9); loss_fn=nn.CrossEntropyLoss()
for ep in range(a.epochs):
    probe.train(); total=0
    for x,y in train:
        x,y=x.to(device),y.to(device)
        with torch.no_grad(): feat=extract(model,x)[0]
        opt.zero_grad(set_to_none=True); loss=loss_fn(probe(feat),y); loss.backward(); opt.step(); total+=loss.item()
    probe.eval(); correct=n=0
    with torch.no_grad():
        for x,y in test:
            y=y.to(device); feat=extract(model,x.to(device))[0]; pred=probe(feat).argmax(1); correct+=(pred==y).sum().item(); n+=y.numel()
    print(f"epoch={ep+1} loss={total/len(train):.4f} test_acc={correct/n:.4f}")
OUT=Path("outputs/05_dinov2"); OUT.mkdir(parents=True,exist_ok=True); torch.save(probe.state_dict(),OUT/"pets_linear_probe.pt")
