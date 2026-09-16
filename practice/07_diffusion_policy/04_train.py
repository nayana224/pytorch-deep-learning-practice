import argparse

import torch
from torch.utils.data import DataLoader

from common import OUT, make_dataset, make_policy


def move_batch_to_device(batch, device):
    return {
        "obs": {
            "image": batch["obs"]["image"].to(device),
            "agent_pos": batch["obs"]["agent_pos"].to(device),
        },
        "action": batch["action"].to(device),
    }


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--max-steps", type=int, default=None)
args = parser.parse_args()

train_dataset = make_dataset()
validation_dataset = train_dataset.get_validation_dataset()

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
)

policy, device = make_policy()
policy.set_normalizer(train_dataset.get_normalizer())

optimizer = torch.optim.AdamW(
    policy.parameters(),
    lr=1e-4,
    betas=(0.95, 0.999),
    eps=1e-8,
    weight_decay=1e-6,
)

step = 0

for epoch in range(args.epochs):
    policy.train()
    train_loss = 0.0
    train_batches = 0

    for batch in train_loader:
        batch = move_batch_to_device(batch, device)

        # Official policy compute_loss implements:
        # observation + noisy action + timestep -> predicted epsilon
        # target epsilon -> MSE
        loss = policy.compute_loss(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_batches += 1
        step += 1

        if args.max_steps is not None and step >= args.max_steps:
            break

    policy.eval()
    validation_losses = []

    with torch.no_grad():
        for batch_index, batch in enumerate(validation_loader):
            batch = move_batch_to_device(batch, device)
            validation_loss = policy.compute_loss(batch)
            validation_losses.append(validation_loss.item())

            if batch_index >= 9:
                break

    mean_train_loss = train_loss / max(train_batches, 1)
    mean_validation_loss = sum(validation_losses) / max(len(validation_losses), 1)

    print(
        f"epoch={epoch + 1} "
        f"train_eps_mse={mean_train_loss:.6f} "
        f"val_eps_mse={mean_validation_loss:.6f}"
    )

    if args.max_steps is not None and step >= args.max_steps:
        break

checkpoint = {
    "policy": policy.state_dict(),
    "normalizer": policy.normalizer.state_dict(),
}

torch.save(checkpoint, OUT / "scaled_policy.pt")
print("saved:", OUT / "scaled_policy.pt")
