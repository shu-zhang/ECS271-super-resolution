import os
import glob
import random
import kagglehub
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from PIL import Image

path = kagglehub.dataset_download("takihasan/div2k-dataset-for-super-resolution")
print (f"Dataset downloaded to: {path}")

class DIV2KDataset(Dataset):
    def __init__(self, root_dir=path, scale=4, mode='train', crop_size=None, transform=None):
        self.root_dir = root_dir
        self.scale = scale
        self.mode = mode
        self.crop_size = crop_size
        self.transform = transform
        
        if mode == 'train':
            self.hr_dir = os.path.join(root_dir, 'Dataset', 'DIV2K_train_HR')
            if scale == 2:
                self.lr_dir = os.path.join(root_dir, 'Dataset', 'DIV2K_train_LR_bicubic', 'X2')
            elif scale == 4:
                self.lr_dir = os.path.join(root_dir, 'Dataset', 'DIV2K_train_LR_bicubic_X4', 'X4')
            else:
                raise ValueError("Scale must be 2 or 4")
        elif mode == 'valid':
            self.hr_dir = os.path.join(root_dir, 'Dataset', 'DIV2K_valid_HR')
            if scale == 2:
                self.lr_dir = os.path.join(root_dir, 'Dataset', 'DIV2K_valid_LR_bicubic', 'X2')
            elif scale == 4:
                self.lr_dir = os.path.join(root_dir, 'Dataset', 'DIV2K_valid_LR_bicubic_X4', 'X4')
            else:
                raise ValueError("Scale must be 2 or 4")
        
            
        self.hr_files = sorted(glob.glob(os.path.join(self.hr_dir, '*.png')))
        self.lr_files = []
        for hr_path in self.hr_files:
            filename = os.path.basename(hr_path)
            file_id = filename.split('.')[0]
            lr_filename = f"{file_id}x{self.scale}.png"
            self.lr_files.append(os.path.join(self.lr_dir, lr_filename))
        
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        hr_path = self.hr_files[idx]
        lr_path = self.lr_files[idx]
        
        hr_image = Image.open(hr_path).convert('RGB')
        lr_image = Image.open(lr_path).convert('RGB')
        
        if self.crop_size:
            # Random crop
            w, h = lr_image.size
            lr_crop_size = self.crop_size // self.scale
            
            if w < lr_crop_size or h < lr_crop_size:
                 pass
            else:
                i = random.randint(0, h - lr_crop_size)
                j = random.randint(0, w - lr_crop_size)
                
                lr_image = F.crop(lr_image, i, j, lr_crop_size, lr_crop_size)
                hr_image = F.crop(hr_image, i * self.scale, j * self.scale, self.crop_size, self.crop_size)
        
        if self.transform:
            hr_image = self.transform(hr_image)
            lr_image = self.transform(lr_image)
            
        return lr_image, hr_image

