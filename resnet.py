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
valid_dataset = DIV2KDataset(scale=4, mode='valid', crop_size=192, transform=transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=10, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = ResNet(scale_factor=4).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 25

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


# Visualize results
model.eval()
with torch.no_grad():
    # Get a batch from validation loader
    val_lr, val_hr = next(iter(valid_loader))
    val_lr = val_lr.to(device)
    
    # Predict
    val_output = model(val_lr)
    
    # Move to CPU for plotting
    val_lr = val_lr.cpu()
    val_hr = val_hr.cpu()
    val_output = val_output.cpu()
    
    # Define zoom parameters (center crop)
    h, w = val_hr.shape[2], val_hr.shape[3]
    cx, cy = h // 2, w // 2
    crop_size = 50  # Size of the crop in HR pixels
    
    # Calculate crop coordinates
    hr_y1, hr_y2 = cx - crop_size // 2, cx + crop_size // 2
    hr_x1, hr_x2 = cy - crop_size // 2, cy + crop_size // 2
    
    # LR coordinates (scale factor 4)
    lr_y1, lr_y2 = hr_y1 // 4, hr_y2 // 4
    lr_x1, lr_x2 = hr_x1 // 4, hr_x2 // 4

    # Plot
    plt.figure(figsize=(15, 10))
    
    # Full images
    plt.subplot(2, 3, 1)
    plt.imshow(val_lr[0].permute(1, 2, 0).numpy())
    plt.title("LR Input")
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(val_output[0].permute(1, 2, 0).clamp(0, 1).numpy())
    plt.title("Model Output")
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.imshow(val_hr[0].permute(1, 2, 0).numpy())
    plt.title("Ground Truth HR")
    plt.axis('off')

    # Zoomed patches
    plt.subplot(2, 3, 4)
    # Display LR patch
    plt.imshow(val_lr[0, :, lr_y1:lr_y2, lr_x1:lr_x2].permute(1, 2, 0).numpy())
    plt.title("LR Patch (Zoomed)")
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(val_output[0, :, hr_y1:hr_y2, hr_x1:hr_x2].permute(1, 2, 0).clamp(0, 1).numpy())
    plt.title("Model Output Patch (Zoomed)")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(val_hr[0, :, hr_y1:hr_y2, hr_x1:hr_x2].permute(1, 2, 0).numpy())
    plt.title("Ground Truth Patch (Zoomed)")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()