#!/usr/bin/python3

from util import *

from torch import tensor

import matplotlib.pyplot as plt; color = "red"
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

## load data

# filenames = ["train.csv", "test.csv"]
# data = {name.split(".")[0] : import_csv(path = ".", name = name)

path = "."
train_data = import_csv(path = path, name = "train.csv")
test_data = import_csv(path = path, name = "test.csv")

# for i in train[0]
train_raw = {param : tensor([i[n] for i in train_data[1:]]) 
         for n, param in enumerate(train_data[0][:3])}
test_raw = {param : tensor([i[n] for i in test_data[1:]]) 
        for n, param in enumerate(test_data[0][:3])}

# a = extract(data.get("train"), 1).unsqueeze(dim=0).t()

## plot data

fig1, ax1 = plt.subplots(2,1,figsize=(16,12))
ax1[0].set_title("train data", color=color)
ax1[0].plot(train_raw.get("T"),train_raw.get("u"), label="u (raw)")
ax1[0].plot(train_raw.get("T"),train_raw.get("y"), label="y (raw)")
ax1[1].set_title("test data", color=color)
ax1[1].plot(test_raw.get("T"),test_raw.get("u"), label="u (raw)")
ax1[1].plot(test_raw.get("T"),test_raw.get("y"), label="y (raw)")
for ax in ax1:
    ax.legend()
    ax.grid()
    ax.tick_params(colors=color)
plt.close(fig1)

## normalize data

def normalize(x):
    # normal = (x - x.mean()) / x.std(unbiased=False)
    normal = x / x.std(unbiased=False)
    return normal.unsqueeze(dim=0).t()

train = dict(map(lambda d: (d[0], normalize(d[1])), train_raw.items()))
test =  dict(map(lambda d: (d[0], normalize(d[1])), test_raw.items()))

## plot data

ax1[0].plot(train.get("u"), label="u (norm)")
ax1[0].plot(train.get("y"), label="y (norm)")
ax1[1].plot(test.get("u"), label="u (norm)")
ax1[1].plot(test.get("y"), label="y (norm)")
for ax in ax1:
    ax.legend()

## prepare data

from torch.utils.data import TensorDataset, DataLoader
train_ds = TensorDataset(train.get("u"),train.get("y"))
train_dl = DataLoader(train_ds, batch_size=len(train_ds))

## init model

(nu, nx, ny) = (int(i) for i in train_data[1][3:])
model = Model(n_x=nx, n_u=nu)

## prepare input vector

# create empty (zero) matrix for possible states
X=torch.zeros((len(train_data),n_x))
# map output column of the train data to first column of the state matrix.
# since x1 = y, means C = [1 0]. just control theory things
X[:,0]=ys[:,0]
# establish gradient descent computable state matrix
X_est=X.requires_grad_()
# extract first row of the states matrix
X0=X_est[0,:]

steps = 20000
lr = 1e-5
dt = 1 # Ts

## init optimizer

from torch import optim

params_net = list(model.parameters())
optimizer = optim.Adam(params_net, lr = lr)

## fit model

for i in range(1, steps + 1):
    # set back gradient for every iteration step
    optimizer.zero_grad()
    
    # perform one-step ahead prediction
    # pipe N_f net through the integrator
    xdot_int= pipe(model, X_est, us, dt)
    
    # Compute fit loss
    error  = xdot_int - X_est
    loss = torch.mean(error**2)*dt

    if not i % 10:
        logging.info(round(loss.item(),2))
        logging.info(i)
    
    # optimization step
    loss.backward()
    # next step 
    optimizer.step()


## evaluate

# predict states with model 
X_sim=evaluate(model,X0,n_x,us,dt)     # here us
# last state is also output
y_ss=X_sim[:,0]
y_ss=y_ss.unsqueeze(dim=0).t()
# add mean error and multiplay by standard output
y_sim=(y_ss*ys.std(unbiased=False))+ys.mean()
# calculate RMS of the resulting output 
# RMS_train=RMSE(y_sim,ys_raw)

# predict states with model 
X_sim_val=evaluate(model,X0,n_x,ut,dt) # here ut
# last state is also output
y_ss_val=X_sim_val[:,0]
y_ss_val=y_ss_val.unsqueeze(dim=0).t()
# add mean error and multiplay by standard output
y_sim_val=(y_ss_val*ys.std(unbiased=False))+ys.mean()
# calculate RMS of the resulting output 
# RMS_test=RMSE(y_sim_val,yt_raw)

## plot results

# show train results
figt, axt = plt.subplots(figsize=(16,8))
plt.plot(y_sim.squeeze().tolist(), 'b', label='Output sim')
plt.plot(ys_raw, 'k', label='Output ')
plt.title('Train Data')
plt.legend();

# show test results
figv, axv = plt.subplots(figsize=(16,8))
plt.plot(y_sim_val.squeeze().tolist(), 'b', label='Output sim')
plt.plot(yt_raw, 'k', label='Output ')
plt.title('Test Data')
plt.legend();

# plt.show()
