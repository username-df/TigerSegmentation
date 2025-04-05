## Project Introduction

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/5e42525b-54a6-4eda-865a-3da02b067953" width="600" style="margin-right: 20px;" />
</div>

Implementation of the [Unet architecture](https://arxiv.org/pdf/1505.04597) for image segmentation, coded using PyTorch. The model performs binary semantic segmentation on images of tigers. 

## Training

The model was trained using a custom dataset that I created by collecting around 400 publicly available images of tigers from Google Images, I then created binary masks for each image by using Adobe photoshop. I used data augmentation (rotating, flipping, etc) on each image/binary mask pair to increase the dataset size to around 800.

## Results

The model works decently well, but oftenly over or under classifies some of the pixels. 

<div style="display: flex;">
  <img src="https://github.com/user-attachments/assets/2283d7a5-bb12-42c1-8214-169f8161862a" width="300" style="margin-right: 20px;" />
  <img src="https://github.com/user-attachments/assets/b28d4b72-adb7-4caf-9a6d-11a49eac99af" width="300" />
  <img src="https://github.com/user-attachments/assets/b9ba35d8-f725-4b5f-bb8a-0e17892c4111" width="300" />
  <img src="https://github.com/user-attachments/assets/2c9c6d65-aad8-46d8-9061-3dea316f541f" width="300" />
  <img src="https://github.com/user-attachments/assets/13882b35-c5df-4c79-a164-826164019f64" width="300" />
</div>
