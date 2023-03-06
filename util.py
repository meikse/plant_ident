#!/usr/bin/python3

from csv import DictReader, writer

import torch
from torch.nn import Module, Sequential, Linear, Sigmoid

from random import random 
from csv import writer, reader

class InputGenerator:
    '''Generator to create (smooth) input data for a potential plant from random
    data '''
    
    def __init__(self, length) -> None:
        '''init and storafe of random data points. y data is random between 0-1.
        x data is randomly incrementing. x and y values are stores in "data" 
        :length: amount of values on the x-axis'''
        x = 0 
        self.data = []
        for _ in range(length):
            y = random()
            x += random() * 2
            self.data.append([x,y])


    def get_data(self):
        '''returns current data list
        :data: x and y values '''
        return self.data


    def subdivide(self, steps):
        '''adds dots on a line between two points in a linear fashion. Input is
        a list with concurrent lists, like [x, y]
        :steps: integer for amount of splits'''
        raw = self.data
        self. data = []
        for i in range(len(raw)):
            if i >= len(raw)-1:
                break
            else:
                xp0 = raw[i][0]
                yp0 = raw[i][1]
                xp1 = raw[i+1][0]
                yp1 = raw[i+1][1]
                
                xd = xp1-xp0
                yd = yp1-yp0
            
                for _ in range(steps):
                    xs = xd / steps
                    ys = yd / steps
                    self.data.append([xp0, yp0])
                    xp0 += xs
                    yp0 += ys


    def gaussian(self, sigma):
        '''1 dimensional filtering of data via gaussian filter
        :sigma: sigma for filtering'''
        from scipy.ndimage import gaussian_filter1d
        y = [i[-1] for i in self.data]
        smooth = gaussian_filter1d(y, sigma=sigma)
        self.data = [[self.data[i][0],smooth[i]] for i in range(len(self.data))]


    def lowpass(self, Ts, Ti, V):
        ''' discrete lowpass filter (pt1) 
        :Ts:  sampling time 
        :T1:  integrator time constant
        :V:   gain
        '''
        y_old=0
        storage=[]
        data = [i[-1] for i in self.data]
        for u in data:
        # $y(k) = \frac{T_1}{T_1+T_s}y(k-1) + \frac{T_s}{T_1+T_s}V\cdot u(k)$
            y = (Ti/(Ti+Ts)) * y_old  + (Ts*V/(Ti+Ts)) * u
            y_old = y
            storage.append(y)
        self.data =[[self.data[i][0],storage[i]] for i in range(len(self.data))]


def export_csv(data, name = "input.csv"):
    ''' exports the generated input "data" into a csv file.
    :name: name of file
    :data: data [[x,y],...]
    '''
    with open(name,"w", newline="") as file:
        csvfile = writer(file, delimiter=",")
        for i in data:
            csvfile.writerow(i)


def import_csv(name = "input.csv"):
    ''' exports the generated input "data" into a csv file.
    :name: name of file
    :data: data [[x,y],...]
    '''
    data = []
    with open(name,"r") as file:
        csv_reader = reader(file, delimiter=",")
        for i in csv_reader:
            i = [float(i) for i in i]
            data.append(i)
    return data


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
    
