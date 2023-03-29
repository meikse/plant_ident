#!/usr/bin/python3

from pathlib import Path

import torch

from random import random 
from csv import writer, reader


def export_csv(data, name): # obsolete, update to import_csv TODO
    ''' exports the generated input "data" into a csv file.
    :name: name of file
    :data: data [[x,y],...]
    '''
    with open(name,"w", newline="") as file:
        csvfile = writer(file, delimiter=",")
        for i in data:
            csvfile.writerow(i)


def import_csv(name, path = "."):
    ''' exports the generated input "data" into a csv file.
    :name: name of file
    :data: data [[x,y],...]
    '''
    name = (Path(path) / name).as_posix()
    data = []
    with open(name,"r") as file:
        csv_reader = reader(file, delimiter=",")
        i = iter(csv_reader)
        header = next(csv_reader)
        for i in csv_reader:
            i = [float(i) for i in i]
            data.append(i)
        data.insert(0,header)
    return data


class InputGenerator:
    '''Generator to create (smooth) input data for a potential plant from random
    data '''
    
    def __init__(self, length) -> None:
        '''init and storage of random data points. y data is random between 0-1.
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

from torch.nn import Module, Sequential, Linear, Sigmoid, ReLU

class Model(Module):
   
    def __init__(self, n_x, n_u):
        '''
        n_u: number of inputs
        n_x: number of states 
        '''
        super(Model, self).__init__()
        n = 40
        self.net = Sequential(Linear(n_x + n_u, n),  
                              Sigmoid(),
                              Linear(n,n),
                              # torch.nn.ELU(),
                              Sigmoid(),
                              Linear(n,n),
                              # torch.nn.ELU(),
                              Sigmoid(),
                              # ReLU(),
                              Linear(n, n_x),        # n_x since states=output
                              )
    

    def forward(self, X,U):
        '''is not called directly, optimizer is calling it implicitly
        :X: state values in form of a Matrix
        :U: input values in form of a Matrix
        :return: evaluation of the net
        '''
        # concatenate inputs and stated for passing them once into the net
        XU = torch.cat((X,U),-1)
        # it does not matter passing them concatenated into the net
        return self.net(XU)


def pipe(model, X, U, dt):
    ''' pipe for the block diagram:

        x_dot_int = x_0 + int(N_f(x,u),dt)
        x = x_dot_int       # next iteration step (loop)

    :model: neural network
    :X      states
    :U:     initial state, X[0]
    :dt:    time constant (integer)

    returns integrated states provided by the neural network
    '''
    # pass u and x_head into N_f (Fig. 1) dot_x_head 
    xhead_i = model(X.float(),U.float())
    # integrate via cumulative sum of all previous estimated x_i
    X_sum = torch.cumsum(xhead_i, dim=0) # integrator
    # adding initial state -> x0
    xdot_int=torch.add(X[0,:], dt*X_sum)
    return xdot_int


def evaluate(model, X0, n_x, U, dt):
    ''' evaluate trained model discretely

    :X0:  row vector of initial states
    :n_x: number of states 
    :us:  is a vector of inputs
    :dt:  size of time step 

    returns output from the feeded neural network + integrator
    '''
    X_list=X0
    X_sim=torch.zeros((U.shape[0],n_x))          # matrix 
    x_sum=X_list                                 # vector
    for i, val in enumerate(U):
        # previous state, new input 
        x_p1 = model(X_list.float(),val.float()) # vector
        # time constant * current state + previous output
        x_sum = dt * x_p1 + x_sum
        # update initial state for the next loop
        X_list=x_sum
        # save state in matrix
        X_sim[i,:]=X_list
    return X_sim

def normalize(x):
    ''' normalizes tensor x and forms it to a vector with shape (len(x),1)
    :x: torch tensor
    '''
    normal = (x - x.mean()) / x.std(unbiased=False)
    return normal.unsqueeze(dim=0).t()

def denormalize(x):
    ''' denormalizes tensor x and forms it to a vector with shape (len(x),1)
    :x: torch tensor
    '''
    normal = (x * x.std(unbiased=False)) + x.mean()
    return normal #.unsqueeze(dim=0).t()


def RMSE(y_sim, y_val):
    ''' root-mean-square error
    :y_sim:  simulated vector
    :y_val:  comparison vector
    '''
    error = y_sim - y_val
    N = y_sim.shape[0]
    return torch.sqrt(((1/N)*torch.sum(error**2)))
