# Visualize results
from matplotlib.dates import num2date
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from dataset import DIV2KDataset
from model_resnet_sr import ResNetSR
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

valid_test_dataset = DIV2KDataset(scale=4, mode='valid', crop_size=None, transform=transform)

valid_dataset, test_dataset = random_split(
    valid_test_dataset, 
    [50, 50],
    generator=torch.Generator().manual_seed(42)
)

# Create DataLoaders
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNetSR(scale_factor=4)
model.load_state_dict(torch.load('model/resnet/resnet_epoch_25.pth', map_location=device))
model.to(device)