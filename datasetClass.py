import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from random import randint

class TigerDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform_input=None, transform_target=None):
        self.input_dir = input_dir
        self.target_dir = target_dir

        self.tf_input = transform_input
        self.tf_target = transform_target

        file_nums = [int(f[0: f.index('.')]) for f in os.listdir(input_dir)]
        file_nums = sorted(file_nums)
        self.file_names = [f"{num}.jpg" if os.path.exists(f"{input_dir}/{num}.jpg") else f"{num}.jpeg" for num in file_nums]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        input_path = os.path.join(self.input_dir, self.file_names[index])
        target_path = os.path.join(self.target_dir, self.file_names[index])

        input_img = Image.open(input_path).convert('RGB')
        target_img = Image.open(target_path).convert('L')


        if self.tf_input and self.tf_target:
            rand = randint(0, 10001)
            
            torch.manual_seed(rand)
            input_img = self.tf_input(input_img)

            torch.manual_seed(rand)
            target_img = self.tf_target(target_img)

        return (input_img, target_img)