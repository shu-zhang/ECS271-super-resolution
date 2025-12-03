import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class SwinTransformerBlock(nn.Module):
    """ Swin Transformer Block (STL).
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity() # Placeholder for DropPath if needed
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if self.attn_mask is None or self.attn_mask.shape[0] != (H // self.window_size) * (W // self.window_size):
                self.attn_mask = self.calculate_mask(self.input_resolution).to(x.device)
            
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class RSTB(nn.Module):
    """ Residual Swin Transformer Block (RSTB).
    
    As described in Figure 2(a) of the paper.
    """
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # Swin Transformer Layers (STL)
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer)

        # Convolutional layer at the end of the block
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        return self.forward_features(x)

    def forward_features(self, x):
        # x shape: (B, C, H, W)
        shortcut = x
        
        # Reshape for Swin Blocks: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = x.shape
        x_reshaped = x.flatten(2).transpose(1, 2)
        
        # Pass through Swin Layers
        x_reshaped = self.residual_group(x_reshaped)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = x_reshaped.transpose(1, 2).view(B, C, H, W)
        
        # Convolution
        x = self.conv(x)
        
        # Residual Connection (Equation 9 in paper)
        return x + shortcut

class Upsample(nn.Sequential):
    """ Upsample module.
    
    Standard pixel-shuffle upsampling.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

class SwinIR(nn.Module):
    r""" SwinIR: Image Restoration Using Swin Transformer
    
    Args:
        img_size (int | tuple(int)): Input image size. Default 64.
        patch_size (int | tuple(int)): Patch size. Default: 1.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Patch embedding dimension. Default: 96.
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        upscale (int): Upscale factor. 2/3/4/8 for image SR, 1 for denoising.
        img_range (float): Image range. 1. or 255.
        upsampler (str): The reconstruction reconstruction module. 'pixelshuffle' / 'pixelshuffledirect' / 'nearest+conv'.
        resi_connection (str): The convolutional block before residual connection. '1conv' / '3conv'.
    """
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=180, depths=[6, 6, 6, 6, 6, 6], num_heads=[6, 6, 6, 6, 6, 6],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='pixelshuffle', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # -------------------------------------------------------------------------
        # 1. Shallow Feature Extraction (Page 2, Eq 1)
        # -------------------------------------------------------------------------
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # -------------------------------------------------------------------------
        # 2. Deep Feature Extraction (Page 2, Eq 2 & 3)
        # -------------------------------------------------------------------------
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(img_size, img_size),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=drop_path_rate,  # no decay used in paper implementation usually
                         norm_layer=norm_layer)
            self.layers.append(layer)
        
        self.norm = norm_layer(self.num_features)

        # Convolution at the end of Deep Feature Extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # variant mentioned in ablation
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # -------------------------------------------------------------------------
        # 3. High Quality Image Reconstruction (Page 3, Eq 4 & 5)
        # -------------------------------------------------------------------------
        if self.upsampler == 'pixelshuffle':
            # For Classical SR
            self.conv_before_upsample = nn.Conv2d(embed_dim, num_feat, 3, 1, 1)
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # For Lightweight SR (skips the conv_before_upsample)
            self.upsample = Upsample(upscale, embed_dim)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'nearest+conv':
            # For Real-World SR
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_hr = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # For Denoising / JPEG Artifact Reduction (No upsampling)
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        x = self.check_image_size(x)
        
        x_size = (x.shape[2], x.shape[3])
        x = self.conv_first(x)
        x_first = x
        
        # Deep Feature Extraction
        for layer in self.layers:
            # Adjust resolution for Swin Layers if input size changed
            # Note: In a full production implementation, you need to handle dynamic input sizes 
            # by padding to multiples of window_size here.
            layer.input_resolution = x_size 
            layer.residual_group.input_resolution = x_size
            for blk in layer.residual_group.blocks:
                blk.input_resolution = x_size
                
            x = layer(x)

        x = self.conv_after_body(x)
        x = x + x_first  # Residual connection (Eq. 4 in paper)
        return x

    def forward(self, x):
        H, W = x.shape[2:]
        # Handle image range and mean
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # Classical SR
            x = self.forward_features(x)
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # Lightweight SR
            x = self.forward_features(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'nearest+conv':
            # Real-world SR
            x = self.forward_features(x)
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # Denoising
            x_first = x
            x = self.forward_features(x)
            x = self.conv_last(x)
            x = x + x_first # Residual learning (Eq. 5 in paper)

        x = x / self.img_range + self.mean
        x = x[:, :, :H*self.upscale, :W*self.upscale]
        return x

# -----------------------------------------------------------------------------
# Usage Examples based on Paper Configurations
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. Classical Image SR (x2)
    # Paper Section 4.1: RSTB=6, STL=6, Window=8, Channel=180, Heads=6
    model_sr = SwinIR(
        upscale=2, 
        in_chans=3, 
        img_size=64, 
        window_size=8,
        img_range=1., 
        depths=[6, 6, 6, 6, 6, 6], 
        embed_dim=180, 
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler='pixelshuffle', 
        resi_connection='1conv'
    )
    
    # Test input
    input_tensor = torch.randn(1, 3, 64, 64)
    output_sr = model_sr(input_tensor)
    print(f"SR Output Shape: {output_sr.shape}") # Should be (1, 3, 128, 128)

    # 2. Lightweight Image SR
    # Paper Section 4.1: RSTB=4, Channel=60
    model_light = SwinIR(
        upscale=2, 
        in_chans=3, 
        img_size=64, 
        window_size=8,
        img_range=1., 
        depths=[6, 6, 6, 6], 
        embed_dim=60, 
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler='pixelshuffledirect', 
        resi_connection='1conv'
    )
    
    # 3. Denoising (Color)
    # Paper Section 4.1: Window=8, Channel=180, Heads=6
    model_dn = SwinIR(
        upscale=1, 
        in_chans=3, 
        img_size=64, 
        window_size=8,
        img_range=1., 
        depths=[6, 6, 6, 6, 6, 6], 
        embed_dim=180, 
        num_heads=[6, 6, 6, 6, 6, 6],
        mlp_ratio=2, 
        upsampler='',  # No upsampling
        resi_connection='1conv'
    )
    output_dn = model_dn(input_tensor)
    print(f"Denoising Output Shape: {output_dn.shape}") # Should be (1, 3, 64, 64)