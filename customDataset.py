import torchvision.transforms as transforms
from datasetClass import TigerDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image

BATCH_SIZE = 8

class PadToSquare:
    def __call__(self, img: Image):
        width, height = img.size
        target_size = 572
        
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
  
        x_offset = (target_size - width) // 2
        y_offset = (target_size - height) // 2
        
        new_img.paste(img, (x_offset, y_offset))
        
        return new_img
        
class adjust_pixels:
    def __call__(self, img):
        img[img <= 0.30] = 0
        img[img > 0.30] = 1
        img = 1 - img
        return img   
    
tf_input = transforms.Compose([
    transforms.RandomRotation(36),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((388,388)),
    PadToSquare(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.40497827529907227, 0.3686119616031647, 0.29055872559547424], 
                        std=[0.25618886947631836, 0.23313170671463013, 0.2274409383535385])
])

tf_target = transforms.Compose([
    transforms.RandomRotation(36),
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((388, 388)),
    transforms.ToTensor(),
    adjust_pixels(),
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