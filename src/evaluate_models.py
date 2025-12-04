import torch
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from dataset import DIV2KDataset
from model_resnet_sr import ResNetSR
from swinir import SwinIR
from model_bicubic import BicubicSR
from utils import calc_psnr
import os
import csv

def evaluate(model_type, weights_path, save_plots=True, output_dir='analysis_results', device='cuda'):
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    transform = transforms.Compose([transforms.ToTensor()])
    # We use crop_size=None to validate on full images
    dataset = DIV2KDataset(scale=4, mode='valid', crop_size=None, transform=transform)

    valid_dataset, test_dataset = random_split(
    dataset, 
    [50, 50],
    generator=torch.Generator().manual_seed(42)
)


    dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Dataset loaded. Found {len(test_dataset)} images.")

    if model_type == 'resnet':
        model = ResNetSR(scale_factor=4).to(device)
    elif model_type == 'swinir':
        # Using parameters from training_swinir.py
        model = SwinIR(
            in_chans=3, 
            upscale=4,
            embed_dim=60,
            depths=4,
            num_heads=3
        ).to(device)
    elif model_type == 'bicubic':
        model = BicubicSR(scale_factor=4).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if weights_path:
        print(f"Loading weights from: {weights_path}")
        try:
            model.load_state_dict(torch.load(weights_path, map_location=device))
        except Exception as e:
            print(f"Error loading weights: {e}")
            return
    else:
        print("No weights path provided. Using initialized model (correct for Bicubic).")

    model.eval()

    psnr_list = []
    inference_times = []

    print("\nStarting evaluation...")
    

    with torch.no_grad():
        for i, (lr, hr) in enumerate(dataloader):
            lr = lr.to(device)
            hr = hr.to(device)

            # Pad input for SwinIR if needed
            if model_type == 'swinir':
                window_size = 8
                _, _, h_old, w_old = lr.shape
                h_pad = (window_size - h_old % window_size) % window_size
                w_pad = (window_size - w_old % window_size) % window_size
                lr = torch.nn.functional.pad(lr, (0, w_pad, 0, h_pad), 'reflect')

            # Measure inference time
            if device.type == 'cuda':
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                sr = model(lr)
                end_event.record()
                torch.cuda.synchronize()
                elapsed_time = start_event.elapsed_time(end_event) # milliseconds
            else:
                start_time = time.time()
                sr = model(lr)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000 # milliseconds

            # Crop output for SwinIR if needed
            if model_type == 'swinir':
                sr = sr[:, :, :h_old * 4, :w_old * 4]

            inference_times.append(elapsed_time)

            # Clamp output to [0, 1] for valid PSNR calculation
            sr = torch.clamp(sr, 0.0, 1.0)

            psnr = calc_psnr(sr, hr)
            psnr_list.append(psnr)

            if save_plots:
                img_lr = lr.cpu()
                img_sr = sr.cpu()
                img_hr = hr.cpu()
                
                # Unpad LR if SwinIR to match original dimensions for display
                if model_type == 'swinir':
                     img_lr = img_lr[:, :, :h_old, :w_old]

                # Define zoom parameters (center crop)
                h, w = img_hr.shape[2], img_hr.shape[3]
                cx, cy = h // 2, w // 2
                crop_size = 50  # Size of the crop in HR pixels
                
                # Calculate crop coordinates
                hr_y1, hr_y2 = cx - crop_size // 2, cx + crop_size // 2
                hr_x1, hr_x2 = cy - crop_size // 2, cy + crop_size // 2
                
                # LR coordinates (scale factor 4)
                lr_y1, lr_y2 = hr_y1 // 4, hr_y2 // 4
                lr_x1, lr_x2 = hr_x1 // 4, hr_x2 // 4

                plt.figure(figsize=(15, 10))
                
                plt.subplot(2, 3, 1)
                plt.imshow(img_lr[0].permute(1, 2, 0).numpy())
                plt.title("LR Input")
                plt.axis('off')
                
                plt.subplot(2, 3, 2)
                plt.imshow(img_sr[0].permute(1, 2, 0).clamp(0, 1).numpy())
                plt.title(f"{model_type} Output")
                plt.axis('off')
                
                plt.subplot(2, 3, 3)
                plt.imshow(img_hr[0].permute(1, 2, 0).numpy())
                plt.title("Ground Truth HR")
                plt.axis('off')

                # Zoomed patches
                plt.subplot(2, 3, 4)
                # Display LR patch
                plt.imshow(img_lr[0, :, lr_y1:lr_y2, lr_x1:lr_x2].permute(1, 2, 0).numpy())
                plt.title("LR Patch (Zoomed)")
                plt.axis('off')

                plt.subplot(2, 3, 5)
                plt.imshow(img_sr[0, :, hr_y1:hr_y2, hr_x1:hr_x2].permute(1, 2, 0).clamp(0, 1).numpy())
                plt.title(f"{model_type} Patch (Zoomed)")
                plt.axis('off')

                plt.subplot(2, 3, 6)
                plt.imshow(img_hr[0, :, hr_y1:hr_y2, hr_x1:hr_x2].permute(1, 2, 0).numpy())
                plt.title("Ground Truth Patch (Zoomed)")
                plt.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(output_dir, f"{model_type}_img_{i}.png")
                plt.savefig(save_path)
                plt.close()

            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(test_dataset)} images...")

    # Save metrics to CSV
    if output_dir:
        metrics_path = os.path.join(output_dir, f"{model_type}_metrics.csv")
        with open(metrics_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Image Index", "PSNR (dB)", "Inference Time (ms)"])
            for idx, (p, t) in enumerate(zip(psnr_list, inference_times)):
                writer.writerow([idx, p, t])
        print(f"Detailed metrics saved to: {metrics_path}")

    avg_psnr = np.mean(psnr_list)
    avg_time = np.mean(inference_times)

    print("\n==================================")
    print(f"Evaluation Results for {model_type.upper()}")
    print("==================================")
    print(f"Model Weights: {weights_path}")
    print(f"Average PSNR:       {avg_psnr:.2f} dB")
    print(f"Avg Inference Time: {avg_time:.2f} ms/image")
    print("==================================")

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    resnet_weights = "model/resnet/resnet_epoch_25.pth"
    swinir_weights = "model/swinir/swinir25.pth"
    
    output_dir = "analysis_results_test"
    os.makedirs(output_dir, exist_ok=True)

    print("Running evaluation for ResNet...")
    evaluate('resnet', resnet_weights, save_plots=True, output_dir=output_dir)

    print("\n" + "-"*50 + "\n")

    print("Running evaluation for SwinIR...")
    evaluate('swinir', swinir_weights, save_plots=True, output_dir=output_dir)

    print("\n" + "-"*50 + "\n")

    print("Running evaluation for Bicubic Interpolation...")
    evaluate('bicubic', None, save_plots=True, output_dir=output_dir)