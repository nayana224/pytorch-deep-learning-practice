from pathlib import Path
import matplotlib.pyplot as plt
import torch

from common import make_dataset, make_policy, OUT, N_OBS_STEPS

ckpt=OUT/"scaled_policy.pt"
if not ckpt.exists(): raise SystemExit("Run 04_train.py first")
ds=make_dataset(); policy,device=make_policy(); state=torch.load(ckpt,map_location=device); policy.load_state_dict(state["policy"]); policy.normalizer.load_state_dict(state["normalizer"]); policy.eval(); sample=ds[0]
obs={k:v[:N_OBS_STEPS].unsqueeze(0).to(device) for k,v in sample["obs"].items()}
preds=[]
with torch.no_grad():
    for _ in range(8): preds.append(policy.predict_action(obs)["action_pred"][0].cpu())
fig,ax=plt.subplots(figsize=(7,6))
for i,p in enumerate(preds): ax.plot(p[:,0],p[:,1],"o-",alpha=.55,label=f"sample {i}" if i<3 else None)
gt=sample["action"]; ax.plot(gt[:,0],gt[:,1],"k--",linewidth=2,label="demonstration")
ax.set_title("stochastic diffusion samples: multimodality / spread"); ax.set_aspect("equal",adjustable="box"); ax.legend(); plt.tight_layout(); plt.savefig(OUT/"06_multimodal_samples.png",dpi=150); plt.show()
spread=torch.stack(preds).std(0).mean().item(); smooth=torch.stack([(p[1:]-p[:-1]).norm(dim=-1).mean() for p in preds]).mean().item(); print("mean sample std:",spread,"mean action-step distance:",smooth)
