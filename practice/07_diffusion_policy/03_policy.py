import torch
from torch.utils.data import DataLoader

from common import make_dataset, make_policy


ds = make_dataset()
policy, device = make_policy()
policy.set_normalizer(ds.get_normalizer())
batch = next(iter(DataLoader(ds, batch_size=2, shuffle=True)))
batch = {
    "obs": {k: v.to(device) for k, v in batch["obs"].items()},
    "action": batch["action"].to(device),
}
policy.train()
loss = policy.compute_loss(batch)
print("obs image:", batch["obs"]["image"].shape)
print("obs agent_pos:", batch["obs"]["agent_pos"].shape)
print("action:", batch["action"].shape)
print("one official-policy epsilon MSE loss:", float(loss))
print("diffusion model parameters:", sum(p.numel() for p in policy.model.parameters()))
print("vision encoder parameters:", sum(p.numel() for p in policy.obs_encoder.parameters()))
