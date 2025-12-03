import torch
import torch.nn as nn

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


