"""ResNet 논문 실습 코드.

Residual learning의 핵심인 F(x), shortcut x, F(x)+x와
plain network 대비 optimization 차이를 확인하기 위한 공부용 코드다.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import PaperCIFAR10
from resnet import make_plain20, make_resnet20


parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=["plain20", "resnet20"], default="resnet20")
parser.add_argument("--max-iter", type=int, default=64000)
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set = PaperCIFAR10(train=True, augment=True)
test_set = PaperCIFAR10(train=False, augment=False)

train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4)

model = make_resnet20() if args.model == "resnet20" else make_plain20()
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
)

step = 0
while step < args.max_iter:
    model.train()

    for images, labels in train_loader:
        if step >= args.max_iter:
            break

        if step == 32000 or step == 48000:
            for group in optimizer.param_groups:
                group["lr"] *= 0.1

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1

        if step % 1000 == 0:
            print(
                f"step={step} "
                f"loss={loss.item():.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.4f}"
            )

output_dir = Path("outputs/01_resnet")
output_dir.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), output_dir / f"{args.model}.pt")
