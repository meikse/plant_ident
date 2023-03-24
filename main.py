#!/usr/bin/python3

from util import *
from pathlib import Path

from torch import tensor

import matplotlib.pyplot as plt

import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)

## load data

PATH = Path(".")
FILENAME = "train.csv"
name = (PATH / FILENAME).as_posix()
train_data = import_csv(name = name)

FILENAME = "test.csv"
name = (PATH / FILENAME).as_posix()
test_data = import_csv(name = name)

## map dict -> tensor

# T = tensor([i[0] for i in train_data])  # sample data
us_raw = tensor([i[1] for i in train_data])
ys_raw = tensor([i[2] for i in train_data])
ut_raw = tensor([i[1] for i in test_data] )
yt_raw = tensor([i[2] for i in test_data] )

## plot data

fig1, ax1 = plt.subplots(figsize=(16,8))
plt.plot(us_raw, label="input  (train)")
plt.plot(ys_raw, label="output (train)")
plt.legend()
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(16,8))
plt.plot(ut_raw, label="input  (test)")
plt.plot(yt_raw, label="output (test)")
plt.legend()
plt.close(fig2)

## prepare data
# no offset from mean value 
us = us_raw / us_raw.std(unbiased=False)
ys = ys_raw / ys_raw.std(unbiased=False)
ut = ut_raw / ut_raw.std(unbiased=False) 
yt = yt_raw / yt_raw.std(unbiased=False)
us = us.unsqueeze(dim=0).t()
ys = ys.unsqueeze(dim=0).t()
ut = ut.unsqueeze(dim=0).t()
yt = yt.unsqueeze(dim=0).t()

## plot data

fig3, ax3 = plt.subplots(figsize=(16,8))
plt.plot(us_raw, label="raw")
plt.plot(us, label="cleaned")
plt.legend()
plt.close(fig1)

fig4, ax4 = plt.subplots(figsize=(16,8))
plt.plot(ys_raw, label="raw")
plt.plot(ys, label="cleaned")
plt.legend()
plt.close(fig2)

## init model

n_x = 2
n_u = 1
model = Model(n_x=n_x, n_u=n_u)

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
