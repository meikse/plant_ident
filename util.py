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

# -----------------------------------------------------------------------------

from torch.nn import Module, Sequential, Linear, Sigmoid

class Logistic(Module):
   
    def __init__(self, n_x, n_u, n_feat=[40,40,40]):
        '''
        n_u: number of inputs
        n_x: number of states 
        n_fest: number of neurons per layer of the state mapping function
        '''
        super(Logistic, self).__init__()
        self.net = Sequential(
            Linear(n_x + n_u, n_feat[0]),  # 2 states, 1 input
            Sigmoid(),
            Linear(n_feat[0], n_feat[1]),
            Sigmoid(),
            Linear(n_feat[1], n_feat[2]),
            Sigmoid(),
            Linear(n_feat[2], n_x),         # n_x since states=output
            )
    
    def forward(self, X,U):
        '''is not called directly, optimizer is calling it implicitly'''
        # concatenate inputs and stated for passing them once into the net
        XU = torch.cat((X,U),-1)
        # i guess it does not matter passing them concatenated into the net
        DX = self.net(XU)

        return DX
    

class INN:
    
    def __init__(self, nn_model):
        # pass net to this class
        self.nn_model = nn_model

    def INN_est(self, X_est,U,dt):

        # pass u and x_head into N_f (Fig. 1) dot_x_head 
        X_est_torch = self.nn_model(X_est.float(),U.float())
        # integrate via cumulative sum of all previous estimated x_i
        X_sum = torch.cumsum(X_est_torch, dim=0) # integrator
        # first row will be extracted
        # x0=X_est[0,:]
        x0=X_est[0]
        # adding initial state -> x_0
        xdot_int=torch.add(x0,dt*X_sum)

        return xdot_int

# def fit(loader, net, loss, num_epochs=1):
#     opt = Adam(net.parameters(), lr=0.01)

#     # empty lists for storage
#     losses = []
#     epochs = []

    
#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch} --> ', end="\t")
#         N = len(loader)
#         for i, (x, y) in enumerate(loader):
#             # Update the weights of the network
#             opt.zero_grad()               # reset gradients 

#             loss_value = loss(net(x.unsqueeze(dim=0)), y)  # forwarding

#             loss_value.backward()         # backwarding
#             opt.step() 
#             # Store training data
#             epochs.append(epoch+i/N)
#             losses.append(loss_value.item())
#         # print('loss: {:.2}'.format(loss_value.item()), end="\n") # wrong just the last
#     return epochs, losses
