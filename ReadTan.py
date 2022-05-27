from wsgiref import headers
import numpy as np
import random
from prettytable import PrettyTable
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl

class GipAprox:
    # coef
    __alpha = 1
    __beta = 1
    __gamma = 1
    __delta = 1

    # depth
    __C = 100

    # consts
    __sigma = 1
    __r = 1

    # time
    __t = 0

    # curve
    __esp1 = 0
    __esp2 = 0
    __esp3 = 0

    # params ampl and depth
    __A = 0
    __B = 0

    __targets = []
    __set = []

    def __init__(self, alpha, beta, gamma, delta, C, esp1, esp2, esp3):
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__C = C
        self.__esp1 = esp1
        self.__esp2 = esp2
        self.__esp3 = esp3
    
    def __E(self, x):
        return np.tanh(self.__esp1*(x + self.__C) + 1) 

    def __S(self, x):
        return np.tanh(self.__esp2*(x + self.__C) + 1) 

    def __Ss(self, t):
        return np.cos(2 * np.pi * self.__t) + 1

    def __G(self, x):
        return np.cos(self.__esp3*(x + self.__C / 2))
    
    def __X(self, A, B, t):
        self.t = t
        return A + B*np.cos(2*np.pi*t)
    
    def __As(self):
        self.__A = -70
    
    def __Bs(self):
        self.__B = -45

    def J(self):
        self.__As()
        self.__Bs()
        A = self.__A
        B = self.__B
        X = self.__X(A, B, self.__t)
        E = self.__E(X)
        S = self.__S(X)
        Ss = self.__Ss(self.__t)
        G = self.__G(X)
        return - self.__beta * np.power(X, 2) - S * self.__gamma * Ss + self.__alpha * E - self.__delta * G

    def GetX(self):
        return self.__X(self.__A, self.__B, self.__t)

    def GetA(self):
        return self.__A
    
    def GetB(self):
        return self.__B
    
    def SetT(self, t):
        self.__t = t
    
    def Geta(self):
        return self.__alpha
    
    def Getb(self):
        return self.__beta
    
    def Getgy(self):
        return self.__gamma
    
    def Getd(self):
        return self.__delta
    
    def GetEsp1(self):
        return self.__esp1
    
    def GetEsp2(self):
        return self.__esp2
    
    def GetEsp3(self):
        return self.__esp3
    
    def GetE(self, x):
        return self.__E(x)
    
    def GetS(self, x):
        return self.__S(x)
    
    def GetSs(self, t):
        return self.__Ss(t)
    
    def GetG(self, x):
        return self.__G(x)
    
    def ReSet(self, a, b, gy, d, esp1, esp2, esp3):
        self.__alpha = a
        self.__beta = b
        self.__gamma = gy
        self.__delta = d
        self.__esp1 = esp1
        self.__esp2 = esp2
        self.__esp3 = esp3
    
    def FindBestStrat(self, iter):
        max = -10000000000
        t = 0.01
        for i in range(0, iter):
            while(t <= 1.000):
                self.SetT(t)
                if max < self.J():
                    max = self.J()
                    print("max ")
                    print(max)
                    print("t ")
                    print(t)
                    print("a")
                    print(self.Geta())
                    print("b")
                    print(self.Getb())
                    print("gy")
                    print(self.Getgy())
                    print("d")
                    print(self.Getd())
                    print("X")
                    print(self.GetX())
                    print("A")
                    print(self.GetA())
                    print("B")
                    print(self.GetB())
                    print("esp1")
                    print(self.GetEsp1())
                    print("esp2")
                    print(self.GetEsp2())
                    print("esp3")
                    print(self.GetEsp3())
                t += 0.01
            if i % 1000 == 0:
                print(i)
            t = 0.001
            a = random.uniform(0.000, 100.000)
            b = random.uniform(0.000, 100.000)
            gy = random.uniform(0.000, 100.000)
            d = random.uniform(0.000, 100.000)
            esp1 = random.uniform(0.000, 10.000)
            esp2 = random.uniform(0.000, 10.000)
            esp3 = random.uniform(0.000, 10.000)
            self.ReSet(a, b, gy, d, esp1, esp2, esp3)
    
    def GenSet(self, n):
        arr = np.zeros((n, 4))
        for i in range(0, n):
            a = random.uniform(0.000, 100.000)
            b = random.uniform(0.000, 1.000)
            gy = random.uniform(0.000, 100.000)
            d = random.uniform(0.000, 1.000)
            esp1 = random.uniform(0.000, 10.000)
            esp2 = random.uniform(0.000, 10.000)
            esp3 = random.uniform(0.000, 10.000)
            self.ReSet(a, b, gy, d, esp1, esp2, esp3)
            t = random.uniform(0.000, 1.000)
            self.SetT(t)
            self.J()
            arr[i][0] = 100
            arr[i][1] = 100 / 2
            arr[i][2] = self.GetA()
            print(arr[i][2])
            arr[i][3] = self.GetB()
            print(arr[i][3])
        return arr
    
    def GenNetSet(self, n):
        self.__set = np.zeros((n, 4))
        self.__targets = np.zeros(n)
        for i in range(0, n):
            a = random.uniform(0.000, 100.000)
            b = random.uniform(0.000, 100.000)
            gy = random.uniform(0.000, 100.000)
            d = random.uniform(0.000, 100.000)
            esp1 = random.uniform(0.000, 10.000)
            esp2 = random.uniform(0.000, 10.000)
            esp3 = random.uniform(0.000, 10.000)
            self.ReSet(a, b, gy, d, esp1, esp2, esp3)
            t = random.uniform(0.000, 1.000)
            self.SetT(t)
            self.J()
            self.__set[i][0] = self.GetE(self.GetX())
            self.__set[i][1] = self.GetS(self.GetX())
            self.__set[i][2] = self.GetSs(t)
            self.__set[i][3] = self.GetG(self.GetX())
            print(self.GetB())
            if self.GetB() < 0 and self.GetB() > -42:
                self.__targets[i] = 1
            else:
                self.__targets[i] = 0
        return self.__set

    def GetTargets(self):
        return self.__targets

# Entry point

if __name__ == "__main__":
    # j = LinAprox(3, 0.015, 0.000025, 0.01, 100)
    # j = LinAprox(100, 0.001, 29, 0.1, 100)
    j = GipAprox(94.1, 16.4, 25.6, 0.3, 26, 2.7, 3, 0.25)
    data = j.GenSet()
    print(tbl(data, headers=["C", "C0", "A", "B"], tablefmt="fancy_grid"))
    datad = j.GenNetSet(10)
    print(tbl(datad, headers=["Ex", "Sx", "St", "Gx"], tablefmt="fancy_grid"))
    print(j.GetTargets())
    j.SetT(0.45)
    print(j.J())

    # j.FindBestStrat(100000)

    # print(j.GetA())
    # print(j.GetB())
    # x(t) graph
    # x = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    # y = np.zeros(11)
    # for i in range(11):
    #     t = j.SetT(x[i])
    #     y[i] = j.GetX()
    # print(y)
    # ys = np.zeros(11)
    # for i in range(11):
    #     j.SetT(x[i])
    #     t = x[i]
    #     ys[i] = j.GetB() * np.cos(2*np.pi*t)
    # print(ys)

    # f = plt.plot(x, y)
    # plt.show()
    # g = plt.plot(x, ys)
    # plt.show()
