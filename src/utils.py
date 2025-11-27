import torch
import math

# --- PSNR ---
def calc_psnr(sr, hr, max_val=1.0):
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * math.log10(max_val) - 10 * math.log10(mse.item())
    return psnr
