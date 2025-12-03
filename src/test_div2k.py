import os
import argparse
import torch
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image

from dataset import DIV2KDataset
from model_resnet_sr import ResNetSR
from utils import calc_psnr

# -----------------------------
# Args
# -----------------------------
parser = argparse.ArgumentParser(description="Evaluate DIV2K validation subset.")
parser.add_argument(
    "--weights",
    "-w",
    default="checkpoints/epoch_25.pth",
    help="Path to the trained model weights.",
)
args = parser.parse_args()

# -----------------------------
# Config
# -----------------------------
SCALE = 4
CROP = 192  # not used because test does not crop
MODEL_PATH = args.weights

root = os.path.dirname(os.path.abspath(__file__))

VAL_HR = f"{root}/data/div2k/Dataset/DIV2K_valid_HR"
VAL_LR = f"{root}/data/div2k/Dataset/DIV2K_valid_LR_bicubic_X4/X4"

os.makedirs("test_results", exist_ok=True)

# -----------------------------
# Load validation dataset
# -----------------------------
dataset = DIV2KDataset(VAL_HR, VAL_LR, crop_size=CROP, scale=SCALE)
# last 50 images of validation set
test_indices = range(len(dataset) - 50, len(dataset))

test_data = [dataset[i] for i in test_indices]

# -----------------------------
# Load model
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model = ResNetSR(scale_factor=SCALE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# -----------------------------
# Test loop
# -----------------------------
psnr_list = []

for idx, (lr, hr) in zip(test_indices, test_data):
    lr = lr.unsqueeze(0).to(device)   # [1, C, H, W]
    hr = hr.unsqueeze(0).to(device)

    with torch.no_grad():
        sr = model(lr)

    psnr = calc_psnr(sr, hr)
    psnr_list.append(psnr)

    # ---------------------------------------
    # Visualization for each image
    # ---------------------------------------
    lr_img = to_pil_image(lr.squeeze(0).cpu())
    sr_img = to_pil_image(sr.squeeze(0).cpu())
    hr_img = to_pil_image(hr.squeeze(0).cpu())

    # center and zoom patch
    H, W = hr_img.size[1], hr_img.size[0]
    cx, cy = H // 2, W // 2
    crop_size = 50

    hr_y1, hr_y2 = cx - crop_size // 2, cx + crop_size // 2
    hr_x1, hr_x2 = cy - crop_size // 2, cy + crop_size // 2

    lr_y1, lr_y2 = hr_y1 // SCALE, hr_y2 // SCALE
    lr_x1, lr_x2 = hr_x1 // SCALE, hr_x2 // SCALE

    hr_crop = hr_img.crop((hr_x1, hr_y1, hr_x2, hr_y2))
    lr_crop = lr_img.crop((lr_x1, lr_y1, lr_x2, lr_y2))
    sr_crop = sr_img.crop((hr_x1, hr_y1, hr_x2, hr_y2))

    # plot
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f"Validation Image #{idx} | PSNR={psnr:.2f} dB", fontsize=14)

    axs[0,0].imshow(lr_img)
    axs[0,0].set_title("LR (Input)")
    axs[0,0].axis("off")

    axs[0,1].imshow(sr_img)
    axs[0,1].set_title("SR (Model Output)")
    axs[0,1].axis("off")

    axs[0,2].imshow(hr_img)
    axs[0,2].set_title("HR (Ground Truth)")
    axs[0,2].axis("off")

    axs[1,0].imshow(lr_crop.resize((crop_size*4, crop_size*4)))
    axs[1,0].set_title("LR Zoomed (upsampled)")
    axs[1,0].axis("off")

    axs[1,1].imshow(sr_crop)
    axs[1,1].set_title("SR Zoomed")
    axs[1,1].axis("off")

    axs[1,2].imshow(hr_crop)
    axs[1,2].set_title("HR Zoomed")
    axs[1,2].axis("off")

    plt.tight_layout()
    save_path = f"test_results/valid_img_{idx}.png"
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[{idx}] PSNR={psnr:.2f} dB | Saved to {save_path}")

# -----------------------------
# Print final result
# -----------------------------
print("\n==========================")
print(" Final Test Summary")
print("==========================")
print(f"Average PSNR (last 50 images): {sum(psnr_list)/len(psnr_list):.2f} dB")
