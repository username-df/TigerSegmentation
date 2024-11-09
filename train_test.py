import torch
from torch import nn
import torch.optim.adagrad
import torch.optim.adagrad
from unetModel import Unet
from customDataset import train_data, val_data, test_data
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Unet()
model = model.to(device)

epochs = 200
accumulation = 8

LR = 1e-3*accumulation
optimizer = torch.optim.SGD(params=model.parameters(), lr=LR, momentum=0.99)
lossfn = nn.CrossEntropyLoss()

# display the loss over epochs
def loss_curves(epochs, train, val):
    xs = range(1, epochs+1)
    plt.plot(xs, train, 'b-', label="Train Loss")
    plt.plot(xs, val, 'b--', label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

train_hst, val_hst = [], []

for epoch in range(epochs):
    #------------- Train ----------------
    train_loss = 0

    for i,(X,y) in enumerate(train_data):
        model.train()

        X = X.to(device)
        y = y.squeeze().long().to(device)
        
        trainprd = model(X)
        
        loss = lossfn(trainprd, y) 
        train_loss += loss.item()

        loss = loss / accumulation
        loss.backward()
        
        if (i+1) % accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

    train_loss /= len(train_data) 
    
    # --------------- Validation ----------------------
    val_loss = 0
    model.eval()

    with torch.inference_mode():
        for X,y in val_data:

            X = X.to(device)
            y = y.squeeze().long().to(device)
            
            valprd = model(X)

            val_loss += lossfn(valprd, y).item()

        val_loss /= len(val_data)

    model.save(file_name=f"saved{epoch}.pth")

    train_hst.append(train_loss)
    val_hst.append(val_loss)

    print(f"---------------- Epoch {epoch} ---------------------")

    print(f"Train Loss: {train_loss:.2f}\n")

    print(f"Validation loss: {val_loss:.2f}\n")

    print()

loss_curves(epochs, train_hst, val_hst)

# --------------- Test ----------------------
test_loss = 0
model.eval()

with torch.inference_mode():
    for X,y in test_data:

        X = X.to(device)
        y = y.squeeze().long().to(device)

        testprd = model(X)

        test_loss += lossfn(testprd, y).item()

    test_loss /= len(test_data)

print(f"---------------- Testing ---------------------")
print(f"Test loss: {test_loss:.2f}\n")