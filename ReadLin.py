from wsgiref import headers
import numpy as np
import random
from scipy import integrate
from prettytable import PrettyTable
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl

class LinAprox:
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
    __targets = []
    __set = []

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
    
    def __As(self):
        self.__A = (self.__alpha*self.__sigma1 - self.__delta*self.__C - self.__gamma*self.__sigma2) / 2*self.__delta
    
    def __Bs(self):
        self.__B = self.__gamma*self.__sigma2 / (6*self.__beta*np.power(np.pi, 2) + 2*self.__delta)

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
    
    def FindAndPut(self):
        pass

# Entry point

if __name__ == "__main__":
    j = LinAprox(0, 0, 0, 0, 0, 0, 0)
    alpha = random.random(0, 1000)
    beta = random.random(0, 1000)
    gamma = random.random(0, 1000)
    delta = random.random(0, 1000)
    С0 = random.randint(20, 100)
    sigma1 = random.random(0, 10)
    sigma2 = random.random(0, 10)
    j.ReSet(alpha, beta, gamma, delta, С0, sigma1, sigma2)

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
