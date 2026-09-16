import torch

from cifar_resnet import plain_cifar, resnet_cifar


if __name__ == "__main__":
    x = torch.randn(4, 3, 32, 32)
    for name, model in [("plain20", plain_cifar(20)), ("resnet20", resnet_cifar(20)), ("resnet56", resnet_cifar(56))]:
        logits, features = model(x, return_features=True)
        print(f"\n{name}")
        print("input :", x.shape)
        for key, value in features.items():
            print(f"{key:7s}:", value.shape)
        print("logits:", logits.shape)
        print("params:", sum(p.numel() for p in model.parameters()))
