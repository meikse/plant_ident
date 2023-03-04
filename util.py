#!/usr/bin/python3

from csv import DictReader, writer

import torch
from torch.nn import Module, Sequential, Linear, Sigmoid


def loadData(path, file):
    with open((path / file).as_posix()) as file:
        csv_data = DictReader(file, delimiter=',')    
        # prepare dict with keys
        columns = {key: [] for key in csv_data.fieldnames[:-2]}
        for row in csv_data:
            for fieldname in csv_data.fieldnames[:-2]: 
                # get row value for key
                value = float(row.get(fieldname))    
                # store it in the dict
                columns.setdefault(fieldname, []).append(value) 
        return columns

# def writeData(path, file):
#     with open((path / file).as_posix(),"w") as file:
#         csv_data = writer(file, delimiter=',')    
#         # prepare dict with keys
#         columns = {key: [] for key in csv_data.fieldnames[:-2]}
#         for row in csv_data:
#             for fieldname in csv_data.fieldnames[:-2]: 
#                 # get row value for key
#                 value = float(row.get(fieldname))    
#                 # store it in the dict
#                 columns.setdefault(fieldname, []).append(value) 
#         return columns

class Logistic(Module):

    def __init__(self):
        super().__init__()
        self.model = Sequential(Linear(1, 10),
                                Sigmoid(),
                                Linear(10,100),
                                Sigmoid(),
                                Linear(100,1)
                                )
        
    def forward(self, xb):
        # here numeric integrator
        return self.model(xb) 


def RMS(y, yhat):
    return torch.sqrt(torch.abs(y - yhat)**2)

def costFunc(xhat_i, xhat, dt, yhat, y, alpha):
    return torch.mean((yhat-y)**2)-(1/alpha)*torch.mean((xhat_i-xhat)**2)*dt
    
