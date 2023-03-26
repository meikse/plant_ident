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
        T = list(range(len(u)))
    
        # # format: s^n ... + s + 1
        # nom=[1]
        # den=[24,4,1]
        # g = tf(nom, den)

        a = [[-1/6, -1/24],[1, 0]]
        b = [[1],[0]]
        c = [1, 0]
        g = ss(a,b,c,0)
        T,y = forced_response(g,T=T, U=u)

        nu = sum([1 for i in b if i[0] != 0])
        nx = len(a)
        ny = sum([1 for i in c if i != 0])

        output_data=[[T[i+1],u[i+1],y[i+1]] for i in range(len(T)-1)]
        output_data.insert(0,["T","u","y","nu","nx","ny"])
        output_data.insert(1,[T[0],u[0],y[0],nu,nx,ny])
        export_csv(output_data, "{}.csv".format(name))
        

if __name__ == "__main__":
    main()
