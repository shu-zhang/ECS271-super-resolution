import torch
import torch.nn as nn
import torch.nn.functional as F

class SwinIR(nn.Module):

    def __init__(self, in_chans=3, embed_dim=96, depths=4, num_layers=4, num_heads=8, window_size=8, upscale=2):

        super().__init__()

        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        self.patch_embed = PatchEmbed(in_chans=embed_dim, embed_dim=embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)

        self.layers = nn.ModuleList([
            RSTB(dim=embed_dim, depth=depths, num_heads=num_heads, window_size=window_size) for _ in range(num_layers)
        ])

        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        self.upsample = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim * (upscale ** 2), 3, 1, 1),
            nn.PixelShuffle(upscale),
            nn.Conv2d(embed_dim, in_chans, 3, 1, 1)
        )

    def forward(self, x):
        x_first = self.conv_first(x)  

        B, C, H, W = x_first.shape
        x_tokens = self.patch_embed(x_first)  
        x_size = (H, W)

        for layer in self.layers:
            x_tokens = layer(x_tokens, x_size)

        x_body_features = self.patch_unembed(x_tokens, x_size)  
        x_body = self.conv_after_body(x_body_features) + x_first
        output = self.upsample(x_body)  

        return output


class SwinBlock(nn.Module):

    def __init__(self, dim=96, input_resolution=None, num_heads=8, window_size=8, mlp_ratio=4.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = window_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, x_size):

        H, W = x_size
        B, L, C = x.shape
        assert L == H * W
        shortcut = x
        x = self.norm1(x)

        x = x.view(B, H, W, C)

        x_windows = window_partition(x, self.window_size)  
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  

        attn_windows = self.attn(x_windows)  
        x = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(x, H, W, self.window_size)  

        x = x.view(B, H * W, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

class RSTB(nn.Module):

    def __init__(self, dim, depth, num_heads=8, window_size=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(
                dim=dim,
                input_resolution=None,
                num_heads=num_heads,
                window_size=window_size
            )
            for _ in range(depth)
        ])
        self.conv_residual = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x, x_size):

        B, L, C = x.shape
        H, W = x_size
        assert L == H * W

        shortcut = x

        for blk in self.blocks:
            x = blk(x, x_size)

        x_img = x.view(B, H, W, C).permute(0, 3, 1, 2)  
        x_conv = self.conv_residual(x_img)  
        x_tokens = x_conv.permute(0, 2, 3, 1).reshape(B, H * W, C)  

        return x_tokens + shortcut

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=1, in_chans=96, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2) 
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        B, N, C = x.shape
        H, W = x_size
        assert N == H * W
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

class WindowAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x)  
        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = F.softmax(attn, dim=-1)

        x = (attn @ v) 
        x = x.transpose(1, 2).reshape(B_, N, C) 

        x = self.proj(x)
        return x


def window_partition(x, window_size):
    B, H, W, C = x.shape
    assert H % window_size == 0 and W % window_size == 0
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, H, W, window_size):
    B_prime, M1, M2, C = windows.shape
    assert M1 == window_size and M2 == window_size
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    B = B_prime // (num_windows_h * num_windows_w)

    x = windows.view(B, num_windows_h, num_windows_w, window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, C)
    return x