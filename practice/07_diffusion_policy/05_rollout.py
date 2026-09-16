from pathlib import Path
import matplotlib.pyplot as plt
import torch

from common import make_dataset, make_policy, OUT, N_OBS_STEPS, N_ACTION_STEPS

ckpt = OUT / "scaled_policy.pt"
if not ckpt.exists(): raise SystemExit("Run 04_train.py first")
ds = make_dataset(); policy, device = make_policy(); state = torch.load(ckpt, map_location=device); policy.load_state_dict(state["policy"]); policy.normalizer.load_state_dict(state["normalizer"]); policy.eval()
sample = ds[0]
obs = {k: v[:N_OBS_STEPS].unsqueeze(0).to(device) for k,v in sample["obs"].items()}
with torch.no_grad(): result = policy.predict_action(obs)
pred = result["action_pred"][0].cpu(); execute = result["action"][0].cpu(); gt = sample["action"]
print("full predicted horizon:", pred.shape, "receding-horizon execution chunk:", execute.shape)
print("policy slices action_pred from start=n_obs_steps-1 for n_action_steps, then replans after execution.")
fig,ax=plt.subplots(figsize=(6,5)); ax.plot(gt[:,0],gt[:,1],"o-",label="demonstration 16-step"); ax.plot(pred[:,0],pred[:,1],"o-",label="predicted 16-step"); ax.plot(execute[:,0],execute[:,1],"o-",linewidth=3,label="execute next 8"); ax.legend(); ax.set_aspect("equal",adjustable="box"); ax.set_title("action sequence + receding horizon chunk"); plt.tight_layout(); plt.savefig(OUT/"05_receding_horizon.png",dpi=150); plt.show()
