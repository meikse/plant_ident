#!/usr/bin/python3

from control import tf
import control
import matplotlib.pyplot as plt

from util import *

from csv import reader

data = []
with open("input.csv","r") as file:
    csv_reader = reader(file, delimiter=",")
    for i in csv_reader:
        data.append(i)

data = data[0]
data = [float(i) for i in data]
# print(data)

# format: s^n ... + s + 1
nom=[1]
den=[1,1,1]
g = tf(nom, den)

n = len(data)
T = [20*i/n for i in range(n)]

# from random import random
# U = [random() for _ in range(len(T))]     # noise

# import math
# U = [math.sin(i) for i in T]              # sin

U = [1 for _ in T]                        # step

plt.plot(T,U, '--')
plt.plot(T,data, '--')
T,x = control.forced_response(g,T=T, U=U)
plt.plot(T,x)
T,x = control.forced_response(g,T=T, U=data)
plt.plot(T,x)
plt.show()
