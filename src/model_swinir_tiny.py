import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Window partition helpers
# -------------------------
def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1)  # (B, num_H, num_W, ws, ws, C)
    return x.reshape(-1, window_size * window_size, C)

def window_reverse(windows, window_size, H, W, B):
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(B, -1, H, W)

# -------------------------
# Window MHSA Block
# -------------------------
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.window_size = window_size
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x):
        win = window_partition(x, self.window_size)
        out, _ = self.attn(win, win, win)
        B, C, H, W = x.shape
        out = window_reverse(out, self.window_size, H, W, B)
        return out

# -------------------------
# Swin Block (simplified)
# -------------------------
class SwinBlock(nn.Module):
    def __init__(self, dim=64, num_heads=4, window_size=8, mlp_ratio=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_perm = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        
        res = x_perm
        x_perm = self.norm1(x_perm)
        x_attn = self.attn(x_perm.permute(0, 3, 1, 2))
        x_perm = res + x_attn.permute(0, 2, 3, 1)

        res2 = x_perm
        x_perm = self.norm2(x_perm)
        x_perm = res2 + self.mlp(x_perm)

        return x_perm.permute(0, 3, 1, 2)  # BHWC -> BCHW

# -------------------------
# Tiny SwinIR model
# -------------------------
class SwinIR_Tiny(nn.Module):
    def __init__(self, dim=64, num_blocks=4, window_size=8, upscale=4):
        super().__init__()
        self.upscale = upscale

        self.entry = nn.Conv2d(3, dim, 3, padding=1)
        self.blocks = nn.Sequential(*[
            SwinBlock(dim=dim, num_heads=4, window_size=window_size)
            for _ in range(num_blocks)
        ])

        self.upsample = nn.Sequential(
            nn.Conv2d(dim, dim * upscale * upscale, 3, padding=1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(dim, 3, 3, padding=1)
        )

    def forward(self, x):
        x = self.entry(x)
        x = self.blocks(x)
        x = self.upsample(x)
        return x
