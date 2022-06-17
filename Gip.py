import numpy as np
import random
from scipy import integrate
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl
from scipy.optimize import differential_evolution as evo

class GipDataSet:
    n = 10
    random.seed(5)
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
    __e1 = 1
    __e2 = 1
    __e3 = 1

    # time
    __t = 0

    # ampl and depth
    __A = 0
    __B = 0

    # arrays
    __ABArray = np.zeros((n, 2))
    __targetArray = np.zeros(n)
    __coefArray = np.zeros((n, 4))

    def __init__(self, n):
        self.n = n
        self.__coefArray = np.zeros((n, 4))
        self.__ABArray = np.zeros((n, 2))
        self.__targetArray = np.zeros(n)

    def __M1(self, t, A, B):
        return -self.__sigma1 * (np.tanh(self.__e1 * ((A + B * np.cos(2*np.pi*t)) + self.__C) + 1)) 

    def __M2(self, t, A, B):
        return (self.__sigma2 * (np.tanh(self.__e2 * ((A + B * np.cos(2*np.pi*t)) + self.__C) + 1)) * (np.cos(2 * np.pi * t) + 1))

    def __M3(self, t, A, B):
        return ((A + B * np.cos(2*np.pi*t))*(A + B * np.cos(2*np.pi*t)))

    def __M4(self, t, A, B):
        return (np.cosh(self.__e3 * ((A + B * np.cos(2*np.pi*t)) + self.__C0)))

    def __J(self, x):
        m1 = self.__M1(x[2], x[0], x[1])
        m2 = self.__M2(x[2], x[0], x[1])
        m3 = self.__M3(x[2], x[0], x[1])
        m4 = self.__M4(x[2], x[0], x[1])
        return m1*self.__alpha + m2*self.__gamma + m3*self.__beta + m4*self.__delta

    def Find(self):
        for i in range(0, self.n):
            print(i)
            max = -9999999999
            self.__alpha = random.random() * 1000.0 + 0.1
            self.__beta = random.random() * 10.0 + 0.1
            self.__gamma = random.random() * 1000.0 + 0.1
            self.__delta = random.random() * 100.0 + 0.1
            self.__sigma1 = random.random() * 10.0 + 0.1
            self.__sigma2 = random.random() * 10.0 + 0.1
            self.__e1 = random.random() * 1.0 + 0.1
            self.__e2 = random.random() * 1.0 + 0.1
            self.__e3 = random.random() * 1.0 + 0.1
            tmp = evo(func = self.__J, bounds=[(-99, -1), (-49, -1), (0, 1)])
            self.__ABArray[i][0] = tmp.x[0]
            self.__ABArray[i][1] = tmp.x[1]
            self.__coefArray[i][0] = self.__alpha
            self.__coefArray[i][1] = self.__beta
            self.__coefArray[i][2] = self.__gamma
            self.__coefArray[i][3] = self.__delta
            if (-self.__ABArray[i][1] >= 10):
                self.__targetArray[i] = 1
            if (-self.__ABArray[i][1] < 10):
                self.__targetArray[i] = 0
            print("A ", self.__ABArray[i][0], "B ", self.__ABArray[i][1])
        return self.__coefArray, self.__ABArray, self.__targetArray


# j = GipDataSet(100)
# coefs, abs, target = j.Find()
# print(coefs)
# print(target)

            

    