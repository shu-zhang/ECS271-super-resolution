import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class DIV2KDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, crop_size=192, scale=4):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.crop_size = crop_size
        self.scale = scale
        
        self.img_names = sorted([f for f in os.listdir(hr_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        hr = Image.open(os.path.join(self.hr_dir, name)).convert("RGB")
        lr_name = name.replace(".png", f"x{self.scale}.png")
        lr = Image.open(os.path.join(self.lr_dir, lr_name)).convert("RGB")

        # random patch
        lr_w, lr_h = lr.size
        lr_ps = self.crop_size // self.scale
        
        x = random.randint(0, lr_w - lr_ps)
        y = random.randint(0, lr_h - lr_ps)

        lr = TF.crop(lr, y, x, lr_ps, lr_ps)
        hr = TF.crop(hr, y * self.scale, x * self.scale, self.crop_size, self.crop_size)

        return TF.to_tensor(lr), TF.to_tensor(hr)