from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import DIV2KDataset

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv2(self.relu(self.conv1(x)))
        out += x
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, scale_factor=4):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.layer1 = self.make_layer(ResNetBlock, 64, 64, num_blocks=5, stride=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 3 * (scale_factor ** 2), kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.scale_factor = scale_factor

    def make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = []
        for _ in range(num_blocks):
            layers.append(block(in_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x = self.layer1(x1)
        x = self.conv2(x) 
        x += x1
        x = self.conv3(x)
        x = self.pixel_shuffle(x)  
        return x

transform = transforms.Compose([
    transforms.ToTensor(),
])


train_dataset = DIV2KDataset(scale=4, mode='train', crop_size=192, transform=transform)
valid_dataset = DIV2KDataset(scale=4, mode='valid', crop_size=None, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=100, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = ResNet(scale_factor=4).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 2

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        lr_imgs = lr_imgs.to(device)
        hr_imgs = hr_imgs.to(device)
        
        # Zero gradientsk
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(lr_imgs)
        loss = criterion(outputs, hr_imgs)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (i+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

print("Training finished.")
