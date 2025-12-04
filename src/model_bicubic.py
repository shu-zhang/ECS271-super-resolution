import torch
import torch.nn as nn
import torch.nn.functional as F

class BicubicSR(nn.Module):
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        # Use bicubic interpolation to upscale the image
        return F.interpolate(x, scale_factor=self.scale_factor, mode='bicubic', align_corners=False)
