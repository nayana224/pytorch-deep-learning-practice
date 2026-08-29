import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# hyperparameter
batch_size = 64
learning_rate = 0.001
epochs = 1

# MNIST를 Tensor로 변환
transform = transforms.ToTensor()

# Dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform,
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# MLP 모델
class MLP(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        x = x.view(x.size(0), -1) # [64, 1, 28, 28] ->[64, 784]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x
    
model = MLP() # parameter에 모두 기울기를 계산해 놓았는가? 

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)

# Training
for epoch in range(epochs):
    
    model.train() # 만든 멤버함수 호출
    
    total_loss = 0.0
    
    for images, labels in train_loader:
        
        outputs = model(images) # forward() 호출
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        
        loss.backward()
        
        print(model.fc1.weight.grad.shape)
        print(model.fc1.bias.grad.shape)
        
        optimizer.step()
        
        total_loss += loss.item()
        
    average_loss = total_loss / len(train_loader)
    
    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss:{average_loss:.4f}"
    )

# Test
model.eval()

correct = 0
total = 0

with torch.no_grad():
    
    for images, labels in test_loader:
        outputs = model(images)
        
        _, predicted = torch.max(outputs, dim=1)
        
        total += labels.size(0)
        
        correct += (predicted == labels).sum().item()
        
accuracy = 100 * correct / total

print(f"Test Accuracy: {accuracy:.2f}%")