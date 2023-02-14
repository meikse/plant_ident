#!/usr/bin/python3

from util import *

from pathlib import Path

from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam 

## load data

PATH = Path(".")
FILENAME = "data.csv"
data = loadData(PATH,FILENAME)

## map dict -> tensor
us, ys, ut, yt = map(tensor,(data['uEst'], # train input
                             data['uVal'], # train output
                             data['yEst'], # test  input
                             data['yVal']))# test  outputtensors 

## init dataset/dataloader
# training dataset
train_ds = TensorDataset((us),(ys)) 
train_dl = DataLoader(train_ds)
# test dataset
test_ds = TensorDataset((ut),(ut)) 
test_dl = DataLoader(test_ds)
## init model
# instantiate model
model = Logistic()

def fit(loader, net, loss, num_epochs=1):
    opt = Adam(net.parameters(), lr=0.01)

    # empty lists for storage
    losses = []
    epochs = []

    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch} --> ', end="\t")
        N = len(loader)
        for i, (x, y) in enumerate(loader):
            # Update the weights of the network
            opt.zero_grad()               # reset gradients 

            loss_value = loss(net(x), y)  # forwarding

            loss_value.backward()         # backwarding
            opt.step() 
            # Store training data
            epochs.append(epoch+i/N)
            losses.append(loss_value.item())
        # print('loss: {:.2}'.format(loss_value.item()), end="\n") # wrong just the last
    return epochs, losses

## fit model
# fit(train_ds, model, costFunc, 2)
