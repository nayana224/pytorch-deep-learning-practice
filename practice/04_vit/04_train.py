import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision import transforms
from vit import vit_b16,vit_tiny16

p=argparse.ArgumentParser(); p.add_argument("--model",choices=["base","tiny"],default="tiny"); p.add_argument("--epochs",type=int,default=20); p.add_argument("--batch-size",type=int,default=64); a=p.parse_args()
size=384 if a.model=="base" else 224
train_tf=transforms.Compose([transforms.Resize((size,size)),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
test_tf=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
train=DataLoader(CIFAR100("data/04_vit",train=True,download=True,transform=train_tf),batch_size=a.batch_size,shuffle=True,num_workers=4)
test=DataLoader(CIFAR100("data/04_vit",train=False,download=True,transform=test_tf),batch_size=a.batch_size,shuffle=False,num_workers=4)
dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=(vit_b16(100,size) if a.model=="base" else vit_tiny16(100,size)).to(dev)
opt=torch.optim.Adam(model.parameters(),lr=3e-4,betas=(0.9,0.999),weight_decay=0.1); loss_fn=nn.CrossEntropyLoss(); OUT=Path("outputs/04_vit"); OUT.mkdir(parents=True,exist_ok=True)
for ep in range(a.epochs):
    model.train(); total=0
    for x,y in train:
        x,y=x.to(dev),y.to(dev); opt.zero_grad(set_to_none=True); loss=loss_fn(model(x),y); loss.backward(); opt.step(); total+=loss.item()
    model.eval(); correct=n=0
    with torch.no_grad():
        for x,y in test:
            y=y.to(dev); pred=model(x.to(dev)).argmax(1); correct+=(pred==y).sum().item(); n+=y.numel()
    print(f"epoch={ep+1} loss={total/len(train):.4f} acc={correct/n:.4f}")
torch.save({"model":model.state_dict(),"variant":a.model,"size":size},OUT/f"{a.model}.pt")
