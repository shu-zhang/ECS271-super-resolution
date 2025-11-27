import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.relu(self.conv1(x)))

class UpsampleBlock(nn.Module):
    def __init__(self, channels=64, scale=4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * (scale * scale), 3, padding=1)
        self.ps = nn.PixelShuffle(scale)

    def forward(self, x):
        return self.ps(self.conv(x))

class ResNetSR(nn.Module):
    def __init__(self, num_blocks=4, scale_factor=4):
        super().__init__()
        self.entry = nn.Conv2d(3, 64, 9, padding=4)
        self.resblocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.upsample = UpsampleBlock(64, scale_factor)
        self.exit = nn.Conv2d(64, 3, 9, padding=4)

    def forward(self, x):
        x = self.entry(x)
        x = self.resblocks(x)
        x = self.upsample(x)
        return self.exit(x)
