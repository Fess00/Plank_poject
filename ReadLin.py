from ctypes.wintypes import DOUBLE
from wsgiref import headers
import numpy as np
import random
from pandas import DataFrame
from scipy import integrate
from prettytable import PrettyTable
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl

class LinAprox:
    n = 100
    random.seed(2)
    # coef
    __alpha = 1
    __beta = 1
    __gamma = 1
    __delta = 1

    # depth
    __C = 100

    # optimal depth

    __C0 = 50

    # consts
    __sigma1 = 1
    __sigma2 = 1

    # time
    __t = 0

    # 
    __A = -55
    __B = -70

    # set and targets
    __targets = np.zeros(n)
    __xParams = np.zeros(n)
    __setParams = np.zeros((100, 6))
    __setAB = np.zeros((100, 2))
    __set = np.zeros((100, 4))

    def __init__(self, alpha, beta, gamma, delta, C0, sigma1, sigma2):
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__C0 = C0
        self.__sigma1 = sigma1
        self._sigma2 = sigma2
    
    def __E(self, x):
        return self.__sigma1 * (x + self.__C)

    def __S(self, x):
        return self.__sigma2 * (x + self.__C)

    def __Ss(self, t):
        return np.cos(2 * np.pi * self.__t) + 1

    def __G(self, x):
        return np.power(x + self.__C0, 2)

    def M1(self, x):
        return self.__alpha * self.__E(x)
    
    def M2(self, x, t):
        return self.__gamma * self.__S(x) * self.__Ss(t)
    
    def M3(self, x):
        return  self.__beta * np.power(x, 2)
    
    def M4(self, x):
        return  self.__delta * self.__G(x)
    
    def __X(self, A, B, t):
        self.t = t
        return A + B*np.cos(2*np.pi*t)
    
    def __As(self, alpha, gamma, delta, sigma1, sigma2):
        return (alpha*sigma1 - delta*self.__C - gamma*sigma2) / 2*delta
    
    def __Bs(self, beta, gamma, delta, sigma2):
        return gamma*sigma2 / (6*beta*np.power(np.pi, 2) + 2*delta)

    def J(self, X, S, Ss, E, G):
        A = self.__A
        B = self.__B
        X = self.__X(A, B, self.__t)
        E = self.__E(X)
        S = self.__S(X)
        Ss = self.__Ss(self.__t)
        G = self.__G(X)
        return - self.__beta * np.power(X, 2) - S * self.__gamma * Ss + self.__alpha * E - self.__delta * G
    
    def ReSet(self, a, b, gy, d, C0, sigma1, sigma2):
        self.__alpha = a
        self.__beta = b
        self.__gamma = gy
        self.__delta = d
        self.__C0 = C0
        self.__sigma1 = sigma1
        self._sigma2 = sigma2
    
    def MakeParamSet(self):
        a = 0.001
        b = 1.0
        for i in range(0, 100):
            alpha = round(random.uniform(a, b), 2)
            beta = round(random.uniform(a, b), 2)
            gamma = round(random.uniform(a, b), 2)
            delta = round(random.uniform(a, b), 2)
            sigma1 = round(random.uniform(a, b), 2)
            sigma2 = round(random.uniform(a, b), 2)
            self.ReSet(alpha, beta, gamma, delta, self.__C0, sigma1, sigma2)
            self.__setParams[i][0] = alpha
            self.__setParams[i][1] = beta
            self.__setParams[i][2] = gamma
            self.__setParams[i][3] = delta
            self.__setParams[i][4] = sigma1
            self.__setParams[i][5] = sigma2
        return self.__setParams
    
    def MakeABSet(self):
        for i in range(0, 100):
            a = self.__As(self.__setParams[i][0], self.__setParams[i][2], self.__setParams[i][3], self.__setParams[i][4], self.__setParams[i][5])
            b = self.__Bs(self.__setParams[i][1], self.__setParams[i][2], self.__setParams[i][3], self.__setParams[i][5])
            self.__setAB[i][0] = a
            self.__setAB[i][1] = b
            g = DataFrame(data=self.__setAB)
        return self.__setAB

# Entry point

if __name__ == "__main__":
    j = LinAprox(0, 0, 0, 0, 0, 0, 0)
    print(j.MakeParamSet())
    print(j.MakeABSet())
    # gamma*sigma2 / (6*beta*np.power(np.pi, 2) + 2*delta)
    print(10*1 / (6*0.001*np.power(np.pi, 2) + 2*0.001))
    # (alpha*sigma1 - delta*self.__C - gamma*sigma2) / 2*delta
    a = (10*1 - 10*100 - 1000*1) / 2*10
    print(a) 
    print(10 - 10*2*(a + 50) / 4*np.power(np.pi, 2)*0.001+10)

    # print(j.GetA())
    # print(j.GetB())
    # # x(t) graph
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
