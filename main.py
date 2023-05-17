#!/usr/bin/env python3

from util import *

from torch import tensor
# from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt 
color = "black"
# plt.style.use('dark_background')

import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

## load data

path = "./data/CascadedTanks/"
# import system data and information
train_data = import_csv(path = path, name = "train.csv")
test_data = import_csv(path = path, name = "test.csv")

# assign systems: input, output, and timestamp
train = {param : tensor([i[n] for i in train_data[1:]]) 
         for n, param in enumerate(train_data[0][:3])}
test = {param : tensor([i[n] for i in test_data[1:]]) 
        for n, param in enumerate(test_data[0][:3])}

## plot data

fig1, ax1 = plt.subplots(2,1,figsize=(16,12))
ax1[0].set_title("train data", color=color)
ax1[0].plot(train["T"],train["u"], label="u (raw)")
ax1[0].plot(train["T"],train["y"], label="y (raw)")
ax1[1].set_title("test data", color=color)
ax1[1].plot(test["T"],test["u"], label="u (raw)")
ax1[1].plot(test["T"],test["y"], label="y (raw)")
for ax in ax1:
    ax.legend()
    ax.grid()
    ax.tick_params(colors=color)
plt.close(fig1)

## normalize data

# extract "T" from it since it lies on the abzyss. only "u" and "y"
train_uy = {key: train[key] for key in ["u", "y"]}
test_uy = {key: test[key] for key in ["u", "y"]}
# merging normalized data in new dict with old one containing "T"
train= train | dict(map(lambda d: (d[0] + "_norm", normalize(d[1])),
                        train_uy.items()))
test = test | dict(map(lambda d: (d[0] + "_norm", normalize(d[1])),
                        test_uy.items()))

## plot data

ax1[0].plot(train["T"],train["u_norm"], label="u (norm)")
ax1[0].plot(train["T"],train["y_norm"], label="y (norm)")
ax1[1].plot(test["T"],test["u_norm"], label="u (norm)")
ax1[1].plot(test["T"],test["y_norm"], label="y (norm)")
for ax in ax1:
    ax.legend()

## init model

# extract info about numbers of input, output and states
(nu, nx, ny) = (int(i) for i in train_data[1][3:])
# create model with system information
model = Model(n_x=nx, n_u=nu)

## prepare data

# create empty (zero) matrix for possible states
X=torch.zeros((len(train["y_norm"]),nx))
# map output column of the train data to first column of the state matrix.
# since x1 = y, means C = [1 0]. just control theory things
X[:,0]=train["y_norm"].squeeze()
X.requires_grad_()

# extract first row of the states matrix
X0=X[0,:]

## DataLoader 

# train_ds = TensorDataset(train["u"], X, train["y"])  # (too slow)
# train_dl = DataLoader(train_ds, batch_size=len(train_ds))

## init optimizer

steps = 15000
lr = 1e-4
dt = 4 # Ts

from torch import optim

params_net = list(model.parameters())
optimizer = optim.Adam(params_net, lr = lr)

## fit model

def fit():
    for i in range(1, steps + 1):
            # set back gradient for every iteration step
            optimizer.zero_grad()
            
            # perform one-step ahead prediction
            # pipe N_f net through the integrator
            xdot_int= pipe(model, X, train["u_norm"], dt)
            
            # Compute fit loss
            error  = xdot_int - X
            loss = torch.mean(error**2)*dt
    
            if not i % 10:
                logging.info(round(loss.item(),2))
                logging.info(i)
            
            # optimization step
            loss.backward()
            # next step 
            optimizer.step()
    
fit()
## evaluate

for data in [train, test]:
    # predict states with model 
    X_sim=evaluate(model,X0,nx,data["u_norm"],dt)
    # last state is also output
    y_ss=X_sim[:,0]
    y_ss=y_ss.unsqueeze(dim=0).t()
    # add mean error and multiplay by standard output
    y_sim=denormalize(y_ss)
    # compute error in respect to reference output
    error = RMSE(y_sim,data["y_norm"])
    # store into dicts
    data.update({"y_sim": y_sim, "error": error})

## plot results
fig2, ax2 = plt.subplots(2,1,figsize=(16,12))
ax2[0].set_title("train data, RMSe: {:.2}".format(train["error"]), color=color)
ax2[0].plot(train["y_sim"].detach(), 'r', label='y (sim)')
ax2[0].plot(train["y_norm"], 'k', label='y (norm)') #why here not train_raw? TODO
ax2[1].set_title("test data, RMSe: {:.2}".format(test["error"]), color=color)
ax2[1].plot(test["y_sim"].detach(), 'r', label='y (sim)')
ax2[1].plot(test["y_norm"], 'k', label='y (norm)')  #why here not test raw ? TODO
for ax in ax2:
    ax.legend()
    ax.grid()
    ax.tick_params(colors=color)
# plt.close(fig2)

##

plt.show()
