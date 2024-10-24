import os
import torch
import torchvision.transforms as tr
from torch.utils.data import Dataset
from PIL import Image

class TigerDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform

        file_nums = [int(f[0: f.index('.')]) for f in os.listdir(input_dir)]
        file_nums = sorted(file_nums)
        self.file_names = [f"{num}.jpeg" for num in file_nums]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.file_names[index])
        target_path = os.path.join(self.target_dir, self.file_names[index])

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('L')

        if self.transform:
            input_img = self.transform(input_img)
            target_img = self.transform(target_img)

        return (input_img, target_img)