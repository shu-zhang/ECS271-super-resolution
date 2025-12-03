import torch
import math
from matplotlib import pyplot as plt

# --- PSNR ---
def calc_psnr(sr, hr, max_val=1.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr


def saveModel(model, path):
    torch.save(model.state_dict(), path)

def visualization(model, loader , device):
        model.eval()
        with torch.no_grad():
            val_lr, val_hr = next(iter(loader))
            val_lr = val_lr.to(device)
            
            val_output = model(val_lr)
            
            val_lr = val_lr.cpu()
            val_hr = val_hr.cpu()
            val_output = val_output.cpu()
            
            h, w = val_hr.shape[2], val_hr.shape[3]
            cx, cy = h // 2, w // 2
            crop_size = 50 
            
            hr_y1, hr_y2 = cx - crop_size // 2, cx + crop_size // 2
            hr_x1, hr_x2 = cy - crop_size // 2, cy + crop_size // 2
            
            lr_y1, lr_y2 = hr_y1 // 4, hr_y2 // 4
            lr_x1, lr_x2 = hr_x1 // 4, hr_x2 // 4

            plt.figure(figsize=(15, 10))
            
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