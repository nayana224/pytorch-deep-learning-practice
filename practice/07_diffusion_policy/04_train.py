import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import make_dataset, make_policy, OUT


def to_device(batch, device):
    return {"obs": {k: v.to(device) for k, v in batch["obs"].items()}, "action": batch["action"].to(device)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10, help="local scaled run; paper/official config uses 3050")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-steps", type=int, default=None)
    args = p.parse_args()
    ds = make_dataset(); val_ds = ds.get_validation_dataset()
    train = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    policy, device = make_policy(); policy.set_normalizer(ds.get_normalizer())
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-4, betas=(0.95, 0.999), eps=1e-8, weight_decay=1e-6)
    step = 0
    for epoch in range(args.epochs):
        policy.train(); total = 0.0
        for batch in train:
            batch = to_device(batch, device); opt.zero_grad(set_to_none=True); loss = policy.compute_loss(batch); loss.backward(); opt.step(); total += loss.item(); step += 1
            if args.max_steps and step >= args.max_steps: break
        policy.eval(); vals=[]
        with torch.no_grad():
            for i,batch in enumerate(val):
                vals.append(policy.compute_loss(to_device(batch,device)).item())
                if i >= 9: break
        print(f"epoch={epoch+1} train_eps_mse={total/max(1,len(train)):.6f} val_eps_mse={sum(vals)/max(1,len(vals)):.6f}")
        if args.max_steps and step >= args.max_steps: break
    torch.save({"policy": policy.state_dict(), "normalizer": policy.normalizer.state_dict()}, OUT / "scaled_policy.pt")
    print("saved:", OUT / "scaled_policy.pt")


if __name__ == "__main__": main()
