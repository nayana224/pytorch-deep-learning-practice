import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cifar_resnet import plain_cifar, resnet_cifar
from importlib.machinery import SourceFileLoader

_data = SourceFileLoader("resnet_data", "practice/01_resnet/01_data.py").load_module()
PaperCIFAR10 = _data.PaperCIFAR10


def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / total, 100.0 * (1.0 - correct / total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["plain", "resnet"], default="resnet")
    p.add_argument("--depth", type=int, default=20)
    p.add_argument("--max-iter", type=int, default=64000)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(PaperCIFAR10(True, True), batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(PaperCIFAR10(False, False), batch_size=256, shuffle=False, num_workers=4)
    model = (resnet_cifar(args.depth) if args.kind == "resnet" else plain_cifar(args.depth)).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    out = Path("outputs/01_resnet")
    out.mkdir(parents=True, exist_ok=True)
    history = []
    step = 0
    while step < args.max_iter:
        model.train()
        for x, y in train_loader:
            if step >= args.max_iter:
                break
            if step in (32000, 48000):
                for g in opt.param_groups:
                    g["lr"] *= 0.1
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            step += 1
            if step % 1000 == 0 or step == args.max_iter:
                test_loss, test_error = evaluate(model, test_loader, device)
                row = {"step": step, "train_loss": loss.item(), "test_loss": test_loss, "test_error": test_error, "lr": opt.param_groups[0]["lr"]}
                history.append(row)
                print(row)
    stem = f"{args.kind}{args.depth}"
    torch.save({"model": model.state_dict(), "kind": args.kind, "depth": args.depth}, out / f"{stem}.pt")
    (out / f"{stem}.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
