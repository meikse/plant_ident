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
        i = [float(i) for i in i]
        data.append(i)

# print(data)

y = [i[1] for i in data]
x = [i[0] for i in data]
# plt.plot(x,y,'x')

# format: s^n ... + s + 1
nom=[1]
den=[1,1,1]
g = tf(nom, den)

n = len(data)
T = [20*i/n for i in range(n)]

from random import random
U = [random() for _ in range(len(T))]     # noise

# import math
# U = [math.sin(i) for i in T]              # sin

U = [1 for _ in T]                        # step

plt.plot(T,U, '--')
plt.plot(T,y, '--')
T,x = control.forced_response(g,T=T, U=U)
plt.plot(T,x)
T,x = control.forced_response(g,T=T, U=y)
plt.plot(T,x)
plt.show()
