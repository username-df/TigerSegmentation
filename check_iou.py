import torch
from unetModel import Unet
from customDataset import test_data
from torchmetrics import JaccardIndex

model = Unet()
model.load(file_name='saved_1241.pth')
model.to("cpu")

jaccard = JaccardIndex(task='binary')
iou = 0

model.eval()
with torch.inference_mode():
    for X,y in test_data:

        X = X
        y = y.squeeze().long()

        testprd = model(X)

        iou += jaccard(testprd.argmax(dim=1), y)
    iou /= len(test_data)
    
print(f"IOU: {iou}")