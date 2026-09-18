"""Diffusion Policy의 학습 loss가 observation-conditioned noise prediction임을 확인한다.

실제 Push-T image / agent position / action chunk를 한 batch 가져와
공식 policy의 compute_loss를 한 번 호출한다.
"""

from torch.utils.data import DataLoader

from common import make_dataset, make_policy


# 1) 공식 Push-T demonstration dataset
train_dataset = make_dataset()
normalizer = train_dataset.get_normalizer()

# 2) image-conditioned diffusion policy
policy, device = make_policy()
policy.set_normalizer(normalizer)

# 3) demonstration batch 하나만 꺼내 data flow를 확인한다.
loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
)
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

# 4) observation을 condition으로 주고 noisy action에서 epsilon을 예측한다.
#    GT는 sampling한 Gaussian noise이고 loss는 MSE다.
policy.train()
loss = policy.compute_loss(batch_on_device)

print("image observations :", tuple(images.shape))
print("agent positions    :", tuple(agent_positions.shape))
print("action sequence    :", tuple(actions.shape))
print("epsilon MSE loss   :", float(loss))
print(
    "diffusion params   :",
    sum(p.numel() for p in policy.model.parameters()),
)
print(
    "vision params      :",
    sum(p.numel() for p in policy.obs_encoder.parameters()),
)
