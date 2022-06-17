from ctypes.wintypes import DOUBLE
from re import S
from ssl import SSL_ERROR_EOF
import time
from wsgiref import headers
import numpy as np
import random
from pandas import DataFrame
from scipy import integrate
from prettytable import PrettyTable
import scipy.optimize 
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl
from scipy import integrate

class LinAprox:
    __n = 10
    random.seed(2)
    __C = 100
    __C0 = 50
    __setParams = np.zeros((__n, 10))
    __set = np.zeros((__n, 13))
    __bounds = [(-100, -1), (-50, -1)]
    def __init__(self):
        pass

    __alpha = 0
    __beta = 0
    __gamma = 0
    __delta = 0
    __sigma1 = 0
    __sigma2 = 0
    __e1 = 0
    __e2 = 0
    __e3 = 0 
    __t = 0
    def __E(self, x, A, B, sigma1, e1):
        return sigma1 * (np.tanh(e1 * (A + B * np.cos(2*np.pi*x)) + self.__C) + 1)

    def __S(self, x, A, B, sigma2, e2):
        return sigma2 * (np.tanh(e2 * (A + B * np.cos(2*np.pi*x)) + self.__C) + 1)

    def __Ss(self, x):
        return np.cos(2 * np.pi * x) + 1

    def __Ss1(self, x):
        return np.cos(2 * np.pi * x) + 1

    def __G(self, x, A, B, e3):
        return np.cosh(e3 * (A + B * np.cos(2*np.pi*x)) + self.__C0)

    #def __G1(self, x, A, B, e3):
    #    return np.cosh(e3 * ((A + B * np.cos(2*np.pi*x)) + self.__C0))

    def M1(self, x, A, B):
        return -self.__E(x, A, B, self.__sigma1, self.__e1)

    def M2(self, x, A, B):
        return (self.__S(x, A, B, self.__sigma2, self.__e2)*self.__Ss(x))

    def M3(self, x, A, B):
        return (A + B * np.cos(2*np.pi*x))

    def M4(self, x, A, B):
        return self.__G(x, A, B, self.__e3)

    def __J(self, x):
        #M1, err = integrate.quad(self.M1, 0, 1, args = (x[0], x[1]))
        #M2, err = integrate.quad(self.M2, 0, 1, args = (x[0], x[1]))
        #M3, err = integrate.quad(self.M3, 0, 1, args = (x[0], x[1]))
        #M4, err = integrate.quad(self.M4, 0, 1, args = (x[0], x[1]))
        M1 = self.M1(x[2], x[0], x[1])
        M2 = self.M2(x[2], x[0], x[1])
        M3 = self.M3(x[2], x[0], x[1])
        M4 = self.M4(x[2], x[0], x[1])
        return self.__alpha * M1 + self.__gamma * M2 + self.__beta * M3 + self.__delta * M4

    def __J1(self, A, B):
        M1, err = integrate.quad(self.M1, 0, 1, args = (A, B))
        M2, err = integrate.quad(self.M2, 0, 1, args = (A, B))
        M3, err = integrate.quad(self.M3, 0, 1, args = (A, B))
        M4, err = integrate.quad(self.M4, 0, 1, args = (A, B))
        return self.__alpha * M1 + self.__gamma * M2 + self.__beta * M3 + self.__delta * M4

    def MakeParamSet(self):
        for i in range(0, self.__n):
            self.__setParams[i][0] = round(random.random() * 300.0 + 0.1, 2) #alpha
            self.__setParams[i][1] = round(random.random() * 0.9 + 0.1, 2) #beta
            self.__setParams[i][2] = round(random.random() * 300.0 + 0.1, 2) #gamma
            self.__setParams[i][3] = round(random.random() * 10.0 + 0.1, 2) #delta
            self.__setParams[i][4] = round(random.random() * 0.9 + 0.1, 2) #sigma 1
            self.__setParams[i][5] = round(random.random() * 0.9 + 0.1, 2) #sigma 2
            self.__setParams[i][6] = round(random.random() * 9.9 + 0.1, 2) #e1
            self.__setParams[i][7] = round(random.random() * 9.9 + 0.1, 2) #e2
            self.__setParams[i][8] = round(random.random() * 9.9 + 0.1, 2) #e3
            self.__setParams[i][9] = round(random.random() * 140 + 1, 2) #e3
        return self.__setParams

    def globalPar(self, alpha, beta, gamma, delta, sigma1, sigma2, e1, e2, e3):
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__sigma1 = sigma1
        self.__sigma2 = sigma2
        self.__e1 = e1
        self.__e2 = e2
        self.__e3 = e3


    def MakeXSetI(self):
        for i in range(0, self.__n):
            alpha = self.__setParams[i][0]
            beta = self.__setParams[i][1]
            gamma = self.__setParams[i][2]
            delta = self.__setParams[i][3]
            sigma1 = self.__setParams[i][4]
            sigma2 = self.__setParams[i][5] #scipy.optimize.differential_evolution
            e1 = self.__setParams[i][6]
            e2 = self.__setParams[i][7]
            e3 = self.__setParams[i][8]
            C0 = self.__setParams[i][9]
            self.__set[i][12] = i
            tmp = -9e+299
            start = time.process_time()
            print(i, start)
            self.globalPar(alpha, beta, gamma, delta, sigma1, sigma2, e1, e2, e3)
            result = scipy.optimize.differential_evolution(func = self.__J, bounds=[(-99, -1), (-49, -1), (0, 1)])
            print(i, result.x, result.fun)
            #J = self.__J1(result.x[0], result.x[1])
            #if(tmp < J):
            #    tmp = J
            self.__set[i][0] = alpha
            self.__set[i][1] = beta
            self.__set[i][2] = gamma
            self.__set[i][3] = delta
            self.__set[i][4] = e1
            self.__set[i][5] = e2
            self.__set[i][6] = e3
            self.__set[i][7] = sigma1
            self.__set[i][8] = sigma2
            self.__set[i][10] = result.x[0]
            self.__set[i][11] = result.x[1]
            
            if(-result.x[1] > 9 and -result.x[1] < 11):
                self.__set[i][9] = 2
            if(-result.x[1] >= 11):
                self.__set[i][9] = 1
            if(-result.x[1] <= 9):
                self.__set[i][9] = 0
        return self.__set

if __name__ == "__main__":
    j = LinAprox()
    j.MakeParamSet()
    data = j.MakeXSetI()

    print(tbl(data, headers=["alpha", "beta", "gamma", "delta", "e1", "e2", "e3", "sigma1", "sigma2", "Target", "A", "B", "i"], tablefmt="fancy_grid"))