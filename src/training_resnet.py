from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import DIV2KDataset
from model_resnet_sr import ResNetSR
import torchvision.transforms as transforms
from utils import saveModel

transform = transforms.Compose([
    transforms.ToTensor(),
])
import torchvision.transforms as transforms

train_dataset = DIV2KDataset(scale=4, mode='train', crop_size=192, transform=transform)
valid_test_dataset = DIV2KDataset(scale=4, mode='valid', crop_size=None, transform=transform)

valid_dataset, test_dataset = random_split(
    valid_test_dataset, 
    [50, 50],
    generator=torch.Generator().manual_seed(42)
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = ResNetSR(scale_factor=4).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 25

training_loss_epoch = []
valid_loss_epoch = []


def train():
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

        training_loss_epoch.append(running_loss / len(train_loader))
        validate(model)
        saveModel(model, "../model/resnet/resnet_epoch_{}.pth".format(epoch + 1))
    print("Training finished.")


def validate(model):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for lr_imgs, hr_imgs in valid_loader:
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            outputs = model(lr_imgs)
            loss = criterion(outputs, hr_imgs)
            total_loss += loss.item()
    valid_loss_epoch.append(total_loss / len(valid_loader))


train()

#plot losses
plt.figure()
plt.plot([i for i in range(1, num_epochs + 1)], training_loss_epoch)
plt.plot([i for i in range(1, num_epochs + 1)], valid_loss_epoch)
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show() 


# Visualize results
model.eval()
with torch.no_grad():
    # Get a batch from validation loader
    val_lr, val_hr = next(iter(test_loader))
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