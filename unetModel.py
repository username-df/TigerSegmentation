import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2)

    def down(self, x):
        x1 = (doubleconv(3, 64))(x)

        x2 = self.maxpool(x1)
        x2 = (doubleconv(64, 128))(x2)

        x3 = self.maxpool(x2)
        x3 = (doubleconv(128, 256))(x3)

        x4 = self.maxpool(x3)
        x4 = (doubleconv(256, 512))(x4)

        x5 = self.maxpool(x4)
        x5 = (doubleconv(512, 1024))(x5)

        return [x1, x2, x3, x4, x5]
    
    def up(self, connect):
        x6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)(connect[4])
        # skip connection from x4  
        skip = crop_skip(connect[3], x6)
              
        x6 = torch.cat(tensors=(x6, skip), dim=1)
        x6 = (doubleconv(1024, 512))(x6)

        x7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)(x6)
        # skip connection from x3 
        skip = crop_skip(connect[2], x7)

        x7 = torch.cat(tensors=(x7, skip), dim=1)
        x7 = (doubleconv(512, 256))(x7)

        x8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)(x7)
        # skip connection from x2
        skip = crop_skip(connect[1], x8)  

        x8 = torch.cat(tensors=(x8, skip), dim=1)
        x8 = (doubleconv(256, 128))(x8)

        x9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)(x8)
        # skip connection from x1
        skip = crop_skip(connect[0], x9)  

        x9 = torch.cat(tensors=(x9, skip), dim=1)
        x9 = (doubleconv(128, 64))(x9)

        # 1x1 Conv to turn 64 channels into 2 classes
        x10 = nn.Conv2d(64, 2, kernel_size=1, stride=1)(x9)
        
        return x10
    
    def forward(self, x):
        skip_connection = self.down(x)
        x = self.up(skip_connection)
        return x

# Conv2d -> ReLU -> Conv2d -> ReLU
def doubleconv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3),
        nn.ReLU(inplace=True)
    )
    return conv

# Crop skip connection so that it can be concatenated with the ConvTranspose2d output
def crop_skip(skip, xn):
    curr_size = skip.size()[2]
    delta = (curr_size - xn.size()[2]) // 2

    cropped = skip[:, :, delta:curr_size-delta, delta:curr_size-delta]
    return cropped