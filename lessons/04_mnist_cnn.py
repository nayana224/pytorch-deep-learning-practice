import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# hypeparameter
batch_size = 64
learning_rate = 0.001
epochs = 1

# MNIST -> Tensor
transform = transforms.ToTensor()

# Dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# CNN Model
class CNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )
        
        self.relu = nn.ReLU()
        
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        
        self.fc = nn.Linear(
            32 * 7 * 7,
            10,
        )
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        x = self.fc(x)
        
        return x

model = CNN()

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# Training
for epoch in range(epochs):
    
    model.train()
    
    total_loss = 0.0
    
    for images, labels in train_loader:
        
        outputs = model(images)
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    average_loss = total_loss / len(train_loader)
    
    print(
        f"Epoch [{epoch + 1}/{epochs}], "
        f"Loss: {average_loss:.4f}"
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