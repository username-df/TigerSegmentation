import torchvision.transforms as transforms
from datasetClass import TigerDataset
from torch.utils.data import DataLoader
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split

BATCH_SIZE = 8

class ZeroPadding:
    def __init__(self, mode, target_size):
        self.mode = mode.upper()
        self.target_size = target_size

    def __call__(self, image):
        image = ImageOps.contain(image, self.target_size)
        
        zeros = Image.new(self.mode, self.target_size)

        x_offset = (self.target_size[0] - image.width) // 2
        y_offset = (self.target_size[1] - image.height) // 2

        zeros.paste(image, (x_offset, y_offset))
            
        return zeros

class adjust_pixels:
    def __call__(self, img):
        img[img <= 0.30] = 0
        img[img > 0.30] = 1
        return img   
    
tf_input = transforms.Compose([
    ZeroPadding(mode="RGB", target_size=(572,572)), 
    transforms.RandomRotation(36),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((572,572)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40497827529907227, 0.3686119616031647, 0.29055872559547424], 
                        std=[0.25618886947631836, 0.23313170671463013, 0.2274409383535385])
])

tf_target = transforms.Compose([
    ZeroPadding(mode="L", target_size=(388,388)), 
    transforms.RandomRotation(36),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((388, 388)),
    transforms.ToTensor(),
    adjust_pixels()
])

dataset = TigerDataset(input_dir="imageData\\inputs", target_dir="imageData\\targets",
                       transform_input=tf_input, transform_target=tf_target)

temp, test_set = train_test_split(dataset, test_size=0.2, random_state=0)
train_set, val_set = train_test_split(temp, test_size=0.25, random_state=0)

train_data = DataLoader(dataset=train_set,
                        batch_size=BATCH_SIZE,
                        shuffle=True)

val_data = DataLoader(dataset=val_set,
                        batch_size=BATCH_SIZE,
                        shuffle=False)

test_data = DataLoader(dataset=test_set,
                            batch_size=BATCH_SIZE,
                            shuffle=False)