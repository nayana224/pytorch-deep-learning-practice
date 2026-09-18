"""Diffusion Policy 실습 파일.

행동 시퀀스의 조건부 diffusion과 receding-horizon 실행을 관찰하기 위한 코드다.
"""

from torch.utils.data import DataLoader

from common import make_dataset, make_policy


# 1. Official Push-T demonstrations
train_dataset = make_dataset()
normalizer = train_dataset.get_normalizer()

# 2. Official image-conditioned diffusion policy
policy, device = make_policy()
policy.set_normalizer(normalizer)

# 3. One demonstration batch
loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))

images = batch["obs"]["image"].to(device)
agent_positions = batch["obs"]["agent_pos"].to(device)
actions = batch["action"].to(device)

batch_on_device = {
    "obs": {
        "image": images,
        "agent_pos": agent_positions,
    },
    "action": actions,
}

# 4. observation + noisy action + timestep -> predicted noise
#    target = sampled Gaussian noise, loss = MSE
policy.train()
loss = policy.compute_loss(batch_on_device)

print("image observations :", images.shape)
print("agent positions    :", agent_positions.shape)
print("action sequence    :", actions.shape)
print("epsilon MSE loss   :", float(loss))
print("diffusion params   :", sum(p.numel() for p in policy.model.parameters()))
print("vision params      :", sum(p.numel() for p in policy.obs_encoder.parameters()))
