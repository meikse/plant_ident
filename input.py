#!/usr/bin/python3

import matplotlib.pyplot as plt
import math
from random import random 

# ------------------------------------------------------------------------------
# generate random data as input

n = 20
u = [random() for _ in range(n)]     # noise

slope = []
for i, val in enumerate(u):
    if i >= len(u)-1:
        break
    y = u[i+1]-u[i] # m = y/1, with x = 1
    slope.append(y)

## 

angle = [abs(round(math.sin(s*math.pi),1))*10 for s in slope]

steps = []
step = .1
for n in angle:
    l = 0
    L=[]
    for i in range(int(n)):
        l += step
        L.append(round(l,2))
    steps.append(L)

b = 0
val = 0
data = []
for i,s in enumerate(slope):
    b = val
    for t in steps[i]:
        val = s * t + b
        data.append(val)

# ------------------------------------------------------------------------------
# plot 

fig, ax = plt.subplots(figsize=(16,6))
plt.grid()
plt.plot([0,120],[0,0],'k')
plt.plot([0,0],[0,1],'k')
plt.plot(data, '-o')
plt.plot( u)

# ------------------------------------------------------------------------------
# smooth data via "scipy" gaussian filtering

from scipy.ndimage import gaussian_filter1d

smooth = gaussian_filter1d(data, sigma=3)
smooth = [i-smooth[0] for i in smooth]

plt.plot(smooth)

# ------------------------------------------------------------------------------
# smooth data via discrete low pass filtering

#init
Ts=1 # sampling time 
T1=5 # integrator time constant
V=1  # gain

#plant 
# $y(k) = \frac{T_1}{T_1+T_s}y(k-1) + \frac{T_s}{T_1+T_s}V\cdot u(k)$
def pt1(u, y_old, Ts = 1, V = 1, T1 = 1):
    return (T1/(T1+Ts)) * y_old  + (Ts*V/(T1+Ts)) * u

#starting condition
y_old=0
lowpassed= []

for u in data:
    y = pt1(u,y_old, Ts, V, T1)
    y_old = y
    lowpassed.append(y)
    
# show result
plt.plot(lowpassed)

# ------------------------------------------------------------------------------
# save data

from csv import writer

with open("input.csv","w", newline="") as file:
    data = writer(file, delimiter=",")
    data.writerow(lowpassed)
