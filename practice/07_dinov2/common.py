"""DINOv2 공식 pretrained feature를 로컬 official checkout에서 불러온다.

네트워크 다운로드는 practice 실행 중에 하지 않는다.
먼저 scripts/setup_dinov2.sh로 official repo와 pretrained weight를 준비한다.
"""

from pathlib import Path

import torch
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = ROOT / "external" / "dinov2"


def load_model(name="dinov2_vits14"):
    """setup 단계에서 준비한 official DINOv2와 cached weight를 사용한다."""
    if not OFFICIAL_REPO.exists():
        raise FileNotFoundError(
            "DINOv2 official repo가 없습니다. 먼저 실행하세요:\n"
            "  bash scripts/setup_dinov2.sh"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # source='local'을 사용해 practice 실행 중 GitHub repo를 다시 받지 않는다.
    # pretrained weight도 setup 단계에서 torch hub cache에 미리 받아둔다.
    model = torch.hub.load(
        str(OFFICIAL_REPO),
        name,
        source="local",
        pretrained=True,
    )
    model = model.to(device)
    model.eval()
    return model, device


def image_transform(size=224):
    """DINOv2 ViT에 넣을 ImageNet-style 입력 전처리."""
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
    """CLS token과 patch token을 분리해 논문 representation을 직접 본다."""
    features = model.forward_features(image_batch)

    class_token = features["x_norm_clstoken"]
    patch_tokens = features["x_norm_patchtokens"]

    return class_token, patch_tokens
