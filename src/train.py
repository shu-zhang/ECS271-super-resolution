import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import DIV2KDataset
from model_resnet_sr import ResNetSR
from model_swinir_tiny import SwinIR_Tiny
from utils import calc_psnr

# -------------------------
# Config
# -------------------------
SCALE = 4
CROP = 192
BATCH = 32
LR = 1e-4
EPOCHS = 30

root = os.path.dirname(os.path.abspath(__file__))
TRAIN_HR = f"{root}/data/div2k/Dataset/DIV2K_train_HR"
TRAIN_LR = f"{root}/data/div2k/Dataset/DIV2K_train_LR_bicubic_X4/X4"
VAL_HR = f"{root}/data/div2k/Dataset/DIV2K_valid_HR"
VAL_LR = f"{root}/data/div2k/Dataset/DIV2K_valid_LR_bicubic_X4/X4"

os.makedirs(f"{root}/checkpoints", exist_ok=True)
os.makedirs(f"{root}/plots", exist_ok=True)

# -------------------------
# Dataset
# -------------------------
train_set = DIV2KDataset(TRAIN_HR, TRAIN_LR, crop_size=CROP, scale=SCALE)
val_set = DIV2KDataset(VAL_HR, VAL_LR, crop_size=CROP, scale=SCALE)

train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1)

# -------------------------
# Model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetSR(scale_factor=SCALE).to(device)
#model = SwinIR_Tiny(upscale=SCALE).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses = []
val_psnrs = []

# -------------------------
# Training
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    running = 0

    for lr_img, hr_img in train_loader:
        lr_img, hr_img = lr_img.to(device), hr_img.to(device)

        sr = model(lr_img)
        loss = criterion(sr, hr_img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    train_losses.append(running / len(train_loader))

    # Validation
    model.eval()
    psnr_sum = 0

    with torch.no_grad():
        for lr_img, hr_img in val_loader:
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            sr = model(lr_img)
            psnr_sum += calc_psnr(sr, hr_img)

    val_psnrs.append(psnr_sum / len(val_loader))

    torch.save(model.state_dict(), f"{root}/checkpoints/epoch_{epoch+1}.pth")
    print(f"Epoch {epoch+1}: Loss={train_losses[-1]:.4f}, PSNR={val_psnrs[-1]:.2f} dB")

# -------------------------
# Plot curves
# -------------------------
epochs = range(1, len(train_losses) + 1)
    
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

# Plot validation PSNR
plt.subplot(1, 2, 2)
plt.plot(epochs, val_psnrs, 'r-', label='Validation PSNR', linewidth=2)
plt.title('Validation PSNR Curve', fontsize=14, fontweight='bold')
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('PSNR (dB)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)

plt.tight_layout()
plt.savefig(f"{root}/plots/training_curves.png", dpi=300, bbox_inches='tight')
print("Saved plots.")