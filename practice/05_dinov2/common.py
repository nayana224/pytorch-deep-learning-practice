import torch
from torchvision import transforms


# DINOv2 전체 pretraining을 다시 구현하지 않는다.
# 논문의 official pretrained model에서 class token / patch token을 직접 본다.


def load_model(name="dinov2_vits14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dinov2", name)
    model = model.to(device)
    model.eval()
    return model, device


def image_transform(size=224):
    return transforms.Compose(
        [
            transforms.Resize(size, antialias=True),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def extract_features(model, image_batch):
    features = model.forward_features(image_batch)

    class_token = features["x_norm_clstoken"]
    patch_tokens = features["x_norm_patchtokens"]

    return class_token, patch_tokens


# Backward-compatible names used by existing scripts.
transform = image_transform
extract = extract_features
