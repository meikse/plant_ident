#!/usr/bin/python3

from control import tf
import control

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16,6))

from util import *

## generate random input data

input = InputGenerator(40)
input.subdivide(8)
input.gaussian(3)

input_data = input.get_data()

# since all data points must be equally spaced only y data can be used
# export_csv(input_data) 

u = [i[1] for i in input_data]
T = [i for i in range(len(u))]

## define plant

# format: s^n ... + s + 1
nom=[1]
den=[24,4,1]
g = tf(nom, den)

## step response

U = [1 for _ in T]                        # step
plt.plot(T,U, '--')
T,x = control.forced_response(g,T=T, U=U)
plt.plot(T,x)

## pipe input data through plant

plt.plot(T,u, '--')
T,y = control.forced_response(g,T=T, U=u)
plt.plot(T,y)
plt.show()

## export data

data=[[T[i],u[i],y[i]] for i in range(len(T))]

# export_csv(data, 'train.csv')
# export_csv(data, 'test.csv')
