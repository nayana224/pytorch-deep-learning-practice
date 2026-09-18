"""공통 환경과 논문별 optional dependency 준비 상태를 빠르게 점검한다."""

import importlib.util
import platform
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def module_status(name: str) -> str:
    return "OK" if importlib.util.find_spec(name) is not None else "MISSING"


def path_status(path: Path) -> str:
    return "OK" if path.exists() else "MISSING"


def main() -> None:
    print("=== System ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    print("\n=== PyTorch ===")
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

        x = torch.rand(3, 3, device="cuda")
        print(f"CUDA tensor test: OK ({x.device})")
    else:
        print("CUDA tensor test: SKIP (CPU practice는 실행 가능)")

    print("\n=== Common Python packages ===")
    common_modules = [
        "numpy",
        "matplotlib",
        "PIL",
        "torchvision",
        "tifffile",
        "sklearn",
        "tqdm",
    ]
    for name in common_modules:
        print(f"{name:16s}: {module_status(name)}")

    print("\n=== Optional paper setups ===")

    # SAM은 pip package + checkpoint + SA-1B subset이 모두 필요하다.
    print("[SAM]")
    print(f"segment_anything : {module_status('segment_anything')}")
    print(f"cv2              : {module_status('cv2')}")
    print(f"pycocotools      : {module_status('pycocotools')}")
    print(
        "checkpoint        :",
        path_status(ROOT / "checkpoints/sam/sam_vit_b_01ec64.pth"),
    )
    print(
        "SA-1B subset      :",
        path_status(ROOT / "data/06_sam/sa1b"),
    )

    # DINOv2는 practice 실행 중 네트워크를 사용하지 않도록 local checkout을 확인한다.
    print("\n[DINOv2]")
    print(
        "official repo     :",
        path_status(ROOT / "external/dinov2"),
    )

    # Diffusion Policy는 공식 repo와 별도 Python dependency가 필요하다.
    print("\n[Diffusion Policy]")
    print(
        "official repo     :",
        path_status(ROOT / "external/diffusion_policy"),
    )
    for name in ["diffusers", "zarr", "numcodecs", "hydra", "einops", "dill"]:
        print(f"{name:16s}: {module_status(name)}")

    print("\nSetup commands:")
    print("  SAM              : bash scripts/setup_sam.sh")
    print("  DINOv2           : bash scripts/setup_dinov2.sh")
    print("  Diffusion Policy : bash scripts/setup_diffusion_policy.sh")


if __name__ == "__main__":
    main()
