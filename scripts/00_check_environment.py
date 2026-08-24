import platform
import sys

import torch


def main() -> None:
    """PyTorch와 CUDA 실행 환경을 확인한다."""
    print("=== System ===")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")

    print("\n=== PyTorch ===")
    print(f"torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        for index in range(device_count):
            print(f"GPU {index}: {torch.cuda.get_device_name(index)}")

        x = torch.rand(3, 3, device="cuda")
        print("\nCUDA tensor test:")
        print(x)
        print(f"device: {x.device}")
    else:
        print("\nCUDA를 사용할 수 없습니다. CPU 실습은 계속 진행할 수 있습니다.")


if __name__ == "__main__":
    main()
