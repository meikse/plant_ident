#!/usr/bin/python3

from control import tf, ss, forced_response
from util import InputGenerator, export_csv


def main():
    for name in ["train", "test"]:
        input = InputGenerator(40)
        input.subdivide(8)
        input.gaussian(3)
        input_data = input.get_data()
    
        u = [i[1] for i in input_data]
        T = [i for i in range(len(u))]
    
        # # format: s^n ... + s + 1
        # nom=[1]
        # den=[24,4,1]
        # g = tf(nom, den)

        # choose a stable one, this one is unstable ! TODO
        a = [[-1/6, -1/24],[1, 0]]
        b = [[1],[0]]
        c = [1, 0]
        # c = [0, 1]
        # c = [0, 1/24]
        g = ss(a,b,c,0)

        T,y = forced_response(g,T=T, U=u)

        output_data=[[T[i],u[i],y[i]] for i in range(len(T))]
        export_csv(output_data, "{}.csv".format(name))
        

if __name__ == "__main__":
    main()
