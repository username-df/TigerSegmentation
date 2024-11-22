import numpy as np
import torch
from torchvision import transforms
from flask import Flask, render_template, request,  url_for
from io import BytesIO
import base64
from PIL import Image
from unetModel import Unet
from customDataset import tf_prdin

app = Flask(__name__)
model = Unet()
model.load(file_name='saved_1241.pth')
model.to('cpu')

@app.route('/')
def start():
    return render_template('webpage.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return 'File not found'
    
    file = request.files['image']

    if file.filename == '':
        return 'No selected file'
    
    output = BytesIO()
    uploaded_img = Image.open(BytesIO(file.read()))
    uploaded_img = uploaded_img.convert('RGB')

    X = tf_prdin(uploaded_img)
    X = X.unsqueeze(0)

    model.eval()
    with torch.inference_mode():
        prd = model(X)

    colours = {
        0: (128, 0, 128),  
        1: (144, 238, 144) 
    }

    mask = prd.squeeze(dim=0)
    mask = mask.permute(1, 2, 0)
    mask = torch.argmax(mask, dim=-1)

    mask_colour = torch.stack([torch.stack([torch.tensor(colours[val.item()]) for val in row]) for row in mask])
    mask_colour = mask_colour.numpy().astype(np.uint8)
    mask_colour = Image.fromarray(mask_colour)
    mask_colour = mask_colour.convert('RGBA')

    mask_colour = np.array(mask_colour)
    alpha = mask_colour[:, :, 3]
    alpha = (alpha * 0.4).astype(np.uint8)  

    mask_colour[:, :, 3] = alpha
    mask_colour = Image.fromarray(mask_colour)

    resized_img = uploaded_img.resize((388, 388))
    resized_img = resized_img.convert('RGBA')
    resized_img.paste(mask_colour, (0, 0), mask_colour)

    orig_w, orig_h = uploaded_img.size
    aspect_ratio = orig_w / orig_h

    new_width = int(aspect_ratio * 388)
    new_height = 388

    final_img = resized_img.resize((new_width, new_height))
    final_img.save(output, format='PNG')
    final_img = (base64.b64encode(output.getvalue())).decode('utf-8')
    output.seek(0)

    return render_template('uploadpage.html', final_img=final_img)

if __name__ == "__main__":
    app.run()
