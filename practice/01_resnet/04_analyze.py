import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from cifar_resnet import plain_cifar, resnet_cifar
from importlib.machinery import SourceFileLoader

_data = SourceFileLoader("resnet_data", "practice/01_resnet/01_data.py").load_module()
PaperCIFAR10 = _data.PaperCIFAR10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["plain20", "resnet20", "resnet56"])
    args = p.parse_args()
    out = Path("outputs/01_resnet")
    fig, ax = plt.subplots(figsize=(7, 4))
    for run in args.runs:
        path = out / f"{run}.json"
        if path.exists():
            h = json.loads(path.read_text())
            ax.plot([r["step"] for r in h], [r["test_error"] for r in h], label=run)
    ax.set_xlabel("iteration")
    ax.set_ylabel("test error (%)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "error_curves.png", dpi=150)

    ckpt_path = out / "resnet20.pt"
    if ckpt_path.exists():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = resnet_cifar(20).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
        x, y = next(iter(DataLoader(PaperCIFAR10(False, False), batch_size=8)))
        with torch.no_grad():
            logits, feats = model(x.to(device), return_features=True)
        print("pred:", logits.argmax(1).cpu().tolist())
        print("gt  :", y.tolist())
        for k, v in feats.items():
            print(k, "std=", float(v.std()), "mean_abs=", float(v.abs().mean()))
    plt.show()


if __name__ == "__main__":
    main()
