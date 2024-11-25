import torch
from torch import nn
import torch.nn.init as init
import os
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Crop skip connection so that it can be concatenated with the ConvTranspose2d output
def crop_skip(skip, xn):
    curr_size = skip.size()[2]
    delta = (curr_size - xn.size()[2]) // 2

    cropped = skip[:, :, delta:curr_size-delta, delta:curr_size-delta]
    return cropped

class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = nn.MaxPool2d(stride=2, kernel_size=2)
        
        # Down convolutions
        self.downconv1 = self.doubleconv(3, 64)
        self.downconv2 = self.doubleconv(64, 128)
        self.downconv3 = self.doubleconv(128, 256)
        self.downconv4 = self.doubleconv(256, 512)
        self.downconv5 = self.doubleconv(512, 1024)

        # Transpose convolutions
        self.transconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.transconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.transconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.transconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        # Up convolutions
        self.upconv1 = self.doubleconv(1024, 512)
        self.upconv2 = self.doubleconv(512, 256)
        self.upconv3 = self.doubleconv(256, 128)
        self.upconv4 = self.doubleconv(128, 64)

        # 1x1 convolution
        self.onebyone = nn.Conv2d(64, 2, kernel_size=1, stride=1)

        self.a_val = 0
        self._initialize_weights()

    # Conv2d -> ReLU -> Conv2d -> ReLU
    def doubleconv(self, in_c, out_c):
        conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3),
            nn.PReLU(num_parameters=out_c, device=device),
            nn.Conv2d(out_c, out_c, kernel_size=3),
            nn.PReLU(num_parameters=out_c, device=device)
        )
        return conv

    def _initialize_weights(self):
        for layer in self.modules():
            if isinstance(layer, nn.PReLU):
               self.a_val = layer.weight.mean().item()
                
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d):
                # Kaiming initialization for layers with PReLU activation
                init.kaiming_uniform_(layer.weight, a=self.a_val, mode='fan_in', nonlinearity='leaky_relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    def down(self, x):
        x1 = self.downconv1(x)

        x2 = self.maxpool(x1)
        x2 = self.downconv2(x2)

        x3 = self.maxpool(x2)
        x3 = self.downconv3(x3)

        x4 = self.maxpool(x3)
        x4 = self.downconv4(x4)

        x5 = self.maxpool(x4)
        x5 = self.downconv5(x5)

        return [x1, x2, x3, x4, x5]
    
    def up(self, connect):
        x6 = self.transconv1(connect[4])
        # skip connection from x4  
        skip = crop_skip(connect[3], x6)
              
        x6 = torch.cat(tensors=(x6, skip), dim=1)
        x6 = self.upconv1(x6)

        x7 = self.transconv2(x6)
        # skip connection from x3 
        skip = crop_skip(connect[2], x7)

        x7 = torch.cat(tensors=(x7, skip), dim=1)
        x7 = self.upconv2(x7)

        x8 = self.transconv3(x7)
        # skip connection from x2
        skip = crop_skip(connect[1], x8)  

        x8 = torch.cat(tensors=(x8, skip), dim=1)
        x8 = self.upconv3(x8)

        x9 = self.transconv4(x8)
        # skip connection from x1
        skip = crop_skip(connect[0], x9)  

        x9 = torch.cat(tensors=(x9, skip), dim=1)
        x9 = self.upconv4(x9)

        # 1x1 Conv to turn 64 channels into 2 classes
        x10 = self.onebyone(x9)
        
        return x10
    
    def forward(self, x):
        skip_connection = self.down(x)
        x = self.up(skip_connection)
        return x

    def save(self, file_name='saved.pth'):
        model_folder_path = './models'

        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        
        torch.save({
            'model_state_dict': self.state_dict(), 
            }, file_name)

    def load(self, folder_name= 'models', file_name='saved.pth'):
        model_folder_path = f'./{folder_name}'
        file_path = os.path.join(model_folder_path, file_name)

        if os.path.exists(file_path):
            load_model = torch.load(file_path, map_location=torch.device('cpu'))

            self.load_state_dict(load_model['model_state_dict'])
            
        else:
            print("No saved model found")