# 1차원 배열
import torch
x = torch.tensor([1.0, 2.0, 3.0, 4.0])

print(x)
print(x.shape)
print(x.dtype)
print("")

# 2차원 배열
x = torch.tensor([
    [1.0, 2.0, 3.0],
    [4.0, 5.0, 6.0]
])

print(x)
print(x.shape)
print("")

# Tensor 연산
import torch

x = torch.tensor([
    [1.0, 2.0]
])

W = torch.tensor([
    [1.0, 3.0],
    [2.0, 4.0]
])

y = x @ W

print("x:", x)
print("W:", W)
print("y:", y)

print("x shape:", x.shape)
print("W shape:", W.shape)
print("y shape:", y.shape)
print("")

# Autograd
import torch
w = torch.tensor(2.0, requires_grad=True)

y = w**2

print("y =", y)

y.backward()

print("dy/dw", w.grad)
print("")

# nn.Linear
import torch
import torch.nn as nn

layer = nn.Linear(3, 2)

x = torch.tensor([
    [1.0, 2.0, 3.0]
])

y = layer(x)

print("y:", y)

print("input shape:", x.shape)
print("output shape:", y.shape)
print("layer.weight:", layer.weight)
print("layer.bias:", layer.bias)
print("layer.weight.shape:", layer.weight.shape)
print("layer.bias.shape", layer.bias.shape)
print("")


# 아주 작은 MLP
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
model = MLP()

print("model:", model)
print("")
        

