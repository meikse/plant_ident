#!/usr/bin/python3

from control import tf, forced_response
from util import InputGenerator, export_csv


def main():
    for name in ["train", "test"]:
        input = InputGenerator(40)
        input.subdivide(8)
        input.gaussian(3)
        input_data = input.get_data()
    
        u = [i[1] for i in input_data]
        T = [i for i in range(len(u))]
    
        # format: s^n ... + s + 1
        nom=[1]
        den=[24,4,1]
        g = tf(nom, den)
    
        T,y = forced_response(g,T=T, U=u)
    
        output_data=[[T[i],u[i],y[i]] for i in range(len(T))]
        export_csv(output_data, "{}.csv".format(name))
        

if __name__ == "__main__":
    main()
