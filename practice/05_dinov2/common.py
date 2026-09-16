import torch
from torchvision import transforms


def load_model(name="dinov2_vits14", device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", name).to(device).eval()
    return model, device


def transform(size=224):
    return transforms.Compose([
        transforms.Resize(size, antialias=True),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])


def extract(model, x):
    f = model.forward_features(x)
    return f["x_norm_clstoken"], f["x_norm_patchtokens"]
