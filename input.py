#!/usr/bin/python3

import matplotlib.pyplot as plt
from random import random 
from csv import writer

class InputGenerator:

    
    def __init__(self, length) -> None:
        x = 0 
        self.data = []
        for _ in range(length):
            y = random()
            x += random() * 2
            self.data.append([x,y])


    def get_data(self):
        return self.data


    def subdivide(self, steps):
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
        from scipy.ndimage import gaussian_filter1d
        y = [i[-1] for i in self.data]
        smooth = gaussian_filter1d(y, sigma=sigma)
        self.data = [[self.data[i][0],smooth[i]] for i in range(len(self.data))]


    def lowpass(self, Ts, Ti, V):
        # $y(k) = \frac{T_1}{T_1+T_s}y(k-1) + \frac{T_s}{T_1+T_s}V\cdot u(k)$
        #init
        # Ts=1 # sampling time 
        # T1=5 # integrator time constant
        # V=1  # gain
        #starting condition
        y_old=0                                 # origin at 0
        storage=[]
        data = [i[-1] for i in self.data]
        for u in data:
            y = (Ti/(Ti+Ts)) * y_old  + (Ts*V/(Ti+Ts)) * u
            y_old = y
            storage.append(y)
        self.data =[[self.data[i][0],storage[i]] for i in range(len(self.data))]

    def export(self, name = "input.csv"):
        with open(name,"w", newline="") as file:
            csvfile = writer(file, delimiter=",")
            for i in self.data:
                csvfile.writerow(i)


test = InputGenerator(20)
fig, ax = plt.subplots(figsize=(16,6))

# data = test.get_data()
# y = [i[1] for i in data]
# x = [i[0] for i in data]
# plt.plot(x,y)

test.subdivide(8)

# data = test.get_data()
# y = [i[1] for i in data]
# x = [i[0] for i in data]
# plt.plot(x,y,'o')

test.gaussian(3)

# data = test.get_data()
# y = [i[1] for i in data]
# x = [i[0] for i in data]
# plt.plot(x,y,'o')

# test.lowpass(1,5,1)

data = test.get_data()
y = [i[1] for i in data]
x = [i[0] for i in data]
plt.plot(x,y,'x')

test.export()
