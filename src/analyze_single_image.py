import torch
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from dataset import DIV2KDataset
from model_resnet_sr import ResNetSR
from swinir import SwinIR
from model_bicubic import BicubicSR
from torch.utils.data import random_split

def crop_and_save_image(image_tensor, output_filename, center_x, center_y, width, height, output_dir='single_image_analysis'):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, output_filename)
    
    if image_tensor.dim() == 4:
        image_tensor = image_tensor.squeeze(0)
        
    _, img_h, img_w = image_tensor.shape
    
    # Calculate top-left corner
    top = int(center_y - height // 2)
    left = int(center_x - width // 2)
    
    # Ensure crop is within bounds
    top = max(0, min(top, img_h - height))
    left = max(0, min(left, img_w - width))
    
    crop = image_tensor[:, top:top+height, left:left+width]
    save_image(crop, save_path)
    print(f"Saved crop: {save_path}")

def analyze_single_image(image_index=37, output_dir='single_image_analysis'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Load Dataset
    print("Loading dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    # We use crop_size=None to validate on full images
    dataset = DIV2KDataset(scale=4, mode='valid', crop_size=None, transform=transform)
    _, dataset = random_split(
    dataset, 
    [50, 50],
    generator=torch.Generator().manual_seed(42)
)
    # Get the 38th image (this is the image we chose to analyse in more detail)
    if image_index >= len(dataset):
        print(f"Error: Image index {image_index} out of range (0-{len(dataset)-1})")
        return

    lr, hr = dataset[image_index]
    
    lr = lr.unsqueeze(0).to(device)
    hr = hr.unsqueeze(0).to(device)

    print(f"Analyzing image index {image_index}...")


    save_image(lr, os.path.join(output_dir, 'original_lr.png'))
    save_image(hr, os.path.join(output_dir, 'original_hr.png'))
    print("Saved original LR and HR images.")

    # 1. Bicubic
    print("Running Bicubic...")
    model_bicubic = BicubicSR(scale_factor=4).to(device)
    model_bicubic.eval()
    with torch.no_grad():
        sr_bicubic = model_bicubic(lr)
        sr_bicubic = torch.clamp(sr_bicubic, 0.0, 1.0)
        save_image(sr_bicubic, os.path.join(output_dir, 'bicubic_output.png'))

    # 2. ResNet
    print("Running ResNet...")
    resnet_weights = "model/resnet/resnet_epoch_25.pth"
    model_resnet = ResNetSR(scale_factor=4).to(device)
    if os.path.exists(resnet_weights):
        print(f"Loading ResNet weights from {resnet_weights}")
        model_resnet.load_state_dict(torch.load(resnet_weights, map_location=device))
    else:
        print(f"Warning: ResNet weights not found at {resnet_weights}")
    
    model_resnet.eval()
    with torch.no_grad():
        sr_resnet = model_resnet(lr)
        sr_resnet = torch.clamp(sr_resnet, 0.0, 1.0)
        save_image(sr_resnet, os.path.join(output_dir, 'resnet_output.png'))

    # 3. SwinIR
    print("Running SwinIR...")
    swinir_weights = "model/swinir/swinir25.pth"
    model_swinir = SwinIR(
        in_chans=3, 
        upscale=4,
        embed_dim=60,
        depths=4,
        num_heads=3
    ).to(device)
    
    if os.path.exists(swinir_weights):
        print(f"Loading SwinIR weights from {swinir_weights}")
        model_swinir.load_state_dict(torch.load(swinir_weights, map_location=device))
    else:
        print(f"Warning: SwinIR weights not found at {swinir_weights}")

    model_swinir.eval()
    with torch.no_grad():
        # Pad input for SwinIR
        window_size = 8
        _, _, h_old, w_old = lr.shape
        h_pad = (window_size - h_old % window_size) % window_size
        w_pad = (window_size - w_old % window_size) % window_size
        lr_padded = torch.nn.functional.pad(lr, (0, w_pad, 0, h_pad), 'reflect')

        sr_swinir = model_swinir(lr_padded)
        
        # Crop output
        sr_swinir = sr_swinir[:, :, :h_old * 4, :w_old * 4]
        
        sr_swinir = torch.clamp(sr_swinir, 0.0, 1.0)
        save_image(sr_swinir, os.path.join(output_dir, 'swinir_output.png'))

    print(f"Analysis complete. Images saved to {output_dir}")

def analyze_line_profile(crop_dir, start_point, end_point, output_dir='single_image_analysis/line_profiles'):
    os.makedirs(output_dir, exist_ok=True)
    
    image_files = [
        ('original_hr_crop.png', 'Original HR'),
        ('bicubic_output_crop.png', 'Interpolation'),
        ('resnet_output_crop.png', 'ResNet'),
        ('swinir_output_crop.png', 'Transformer'),
        ('original_lr_crop.png', 'Original LR')
    ]
    
    colors = ['k', 'b', 'g', 'r', 'm']
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(2, 5)
    
    ax_imgs = [
        fig.add_subplot(gs[0, 0]), # HR
        fig.add_subplot(gs[0, 1]), # Bicubic
        fig.add_subplot(gs[0, 2]), # ResNet
        fig.add_subplot(gs[0, 3]), # SwinIR
        fig.add_subplot(gs[0, 4])  # LR
    ]
    
    ax_profile = fig.add_subplot(gs[1, :])
    
    font_size = 18
    
    # Calculate line length and points
    x0, y0 = start_point
    x1, y1 = end_point
    length = int(np.hypot(x1 - x0, y1 - y0))
    num_points = length  # One sample per pixel unit roughly
    
    # Generate sampling points in HR coordinates
    x_samples = np.linspace(x0, x1, num_points)
    y_samples = np.linspace(y0, y1, num_points)
    distance = np.linspace(0, length, num_points)
    
    for (filename, label), color, ax in zip(image_files, colors, ax_imgs):
        path = os.path.join(crop_dir, filename)
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            ax.axis('off')
            continue
            
        # Load image
        img_pil = Image.open(path)
        img_gray = img_pil.convert('L')
        img_arr = np.array(img_gray) / 255.0
        
        # Plot image with line
        ax.imshow(img_pil)
        ax.set_title(label, fontsize=font_size)
        ax.axis('off')
        
        # Determine coordinates for plotting line and sampling
        if filename == 'original_lr_crop.png':
            # LR image is 1/4 size
            scale = 0.25
            ax.plot([x0 * scale, x1 * scale], [y0 * scale, y1 * scale], 'r-', linewidth=2)
            
            # Sample from LR image using scaled coordinates
            # We use the same x_samples, y_samples (HR coords) but scale them for sampling
            samp_x = x_samples * scale
            samp_y = y_samples * scale
        else:
            scale = 1.0
            ax.plot([x0, x1], [y0, y1], 'r-', linewidth=2)
            samp_x = x_samples
            samp_y = y_samples
            
        # Extract profile using bilinear interpolation
        profile = []
        h, w = img_arr.shape
        for sx, sy in zip(samp_x, samp_y):
            sx = np.clip(sx, 0, w - 1.001) # Avoid index out of bounds
            sy = np.clip(sy, 0, h - 1.001)
            
            x_low = int(sx)
            x_high = x_low + 1
            y_low = int(sy)
            y_high = y_low + 1
            
            wx = sx - x_low
            wy = sy - y_low
            
            val = (1 - wx) * (1 - wy) * img_arr[y_low, x_low] + \
                  wx * (1 - wy) * img_arr[y_low, x_high] + \
                  (1 - wx) * wy * img_arr[y_high, x_low] + \
                  wx * wy * img_arr[y_high, x_high]
            profile.append(val)
            
        # Plot profile
        ax_profile.plot(distance, profile, label=label, color=color, linewidth=2)
        
    ax_profile.set_title('Luminescence Profile', fontsize=font_size)
    ax_profile.set_xlabel('Distance (pixels)', fontsize=font_size)
    ax_profile.set_ylabel('Normalized Intensity', fontsize=font_size)
    ax_profile.legend(fontsize=font_size)
    ax_profile.tick_params(axis='both', which='major', labelsize=font_size-2)
    ax_profile.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'line_profile_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved line profile analysis to {save_path}")
    plt.show()

def create_comparison_plot(output_dir='single_image_analysis/crops', save_name='comparison_plot.png'):
    top_images = [
        ('original_hr_crop.png', 'Original HR'),
        ('original_lr_crop.png', 'Original LR')
    ]
    bottom_images = [
        ('bicubic_output_crop.png', 'Interpolation'),
        ('resnet_output_crop.png', 'ResNet'),
        ('swinir_output_crop.png', 'Transformer')
    ]
    
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 6)
    
    # Top row (centered)
    ax1 = fig.add_subplot(gs[0, 1:3])
    ax2 = fig.add_subplot(gs[0, 3:5])
    top_axes = [ax1, ax2]
    
    # Bottom row
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4 = fig.add_subplot(gs[1, 2:4])
    ax5 = fig.add_subplot(gs[1, 4:6])
    bottom_axes = [ax3, ax4, ax5]
    
    font_size = 24

    for ax, (filename, title) in zip(top_axes, top_images):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=font_size)
            ax.axis('off')
        else:
            print(f"Warning: {path} not found")
            ax.axis('off')

    for ax, (filename, title) in zip(bottom_axes, bottom_images):
        path = os.path.join(output_dir, filename)
        if os.path.exists(path):
            img = Image.open(path)
            ax.imshow(img)
            ax.set_title(title, fontsize=font_size)
            ax.axis('off')
        else:
            print(f"Warning: {path} not found")
            ax.axis('off')
            
    plt.tight_layout()
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {save_path}")


if __name__ == "__main__":
    analyze_single_image(image_index=37, output_dir='single_image_analysis')

    from torchvision.io import read_image

    def load_tensor_image(path):
        return read_image(path).float() / 255.0

    hr = load_tensor_image(os.path.join('single_image_analysis', 'original_hr.png'))
    lr = load_tensor_image(os.path.join('single_image_analysis', 'original_lr.png'))
    sr_bicubic = load_tensor_image(os.path.join('single_image_analysis', 'bicubic_output.png'))
    sr_resnet = load_tensor_image(os.path.join('single_image_analysis', 'resnet_output.png'))
    sr_swinir = load_tensor_image(os.path.join('single_image_analysis', 'swinir_output.png'))


    _, h_hr, w_hr = hr.shape
    cx, cy = w_hr * 0.27, h_hr * 0.46
    crop_w, crop_h = 200, 200
    
    print(f"Saving crops with center ({cx}, {cy}) and size ({crop_w}, {crop_h})...")
    output_dir = 'single_image_analysis/crops'
    crop_and_save_image(hr, 'original_hr_crop.png', cx, cy, crop_w, crop_h, output_dir)
    crop_and_save_image(sr_bicubic, 'bicubic_output_crop.png', cx, cy, crop_w, crop_h, output_dir)
    crop_and_save_image(sr_resnet, 'resnet_output_crop.png', cx, cy, crop_w, crop_h, output_dir)
    crop_and_save_image(sr_swinir, 'swinir_output_crop.png', cx, cy, crop_w, crop_h, output_dir)
    crop_and_save_image(lr, 'original_lr_crop.png', cx//4, cy//4, crop_w//4, crop_h//4, output_dir)

    create_comparison_plot(output_dir)

    # Analyze line profile
    # Define start and end points for the line (in HR crop coordinates, 0-200)
    start_point = (55, 78)
    end_point = (92, 31)
    print(f"Analyzing line profile from {start_point} to {end_point}...")
    analyze_line_profile(output_dir, start_point, end_point)
