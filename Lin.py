import numpy as np
import random
from scipy import integrate
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl

class LinDataSet:
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
        self.__ABArray = np.zeros((n, 2))
        self.__targetArray = np.zeros(n)
        self.__coefArray = np.zeros((n, 4))

    def __M1(self, t, A, B, sigma1):
        return sigma1 * ((A + B * np.cos(2*np.pi*t)) + self.__C) # здесь С положительное

    def __M2(self, t, A, B, sigma2):
        return -(sigma2 * ((A + B * np.cos(2*np.pi*t)) + self.__C) * (np.cos(2 * np.pi * t) + 1))

    def __M3(self, t, A, B):
        return -((A + B * np.cos(2*np.pi*t))*(A + B * np.cos(2*np.pi*t)))

    def __M4(self, t, A, B):
        return -np.power((A + B * np.cos(2*np.pi*t)) + self.__C0, 2) # как и здесь С0 положительное 

    def Ms(self, A, B, sigma1, sigma2):
        y1, err1 = integrate.quad(self.__M1, 0, 1, args=(A, B, sigma1))
        y2, err2 = integrate.quad(self.__M2, 0, 1, args=(A, B, sigma2))
        y3, err3 = integrate.quad(self.__M3, 0, 1, args=(A, B))
        y4, err4 = integrate.quad(self.__M4, 0, 1, args=(A, B))
        return y1, y2, y3, y4

    def J(self, M1, M2, M3, M4):
        return M1*self.__alpha + M2*self.__gamma + M3*self.__beta + M4*self.__delta

    def Find(self):
        for i in range(0, self.n):
            print(i)
            max = -9999999999
            self.__alpha = random.random() * 1000.0 + 0.1
            self.__beta = random.random() * 1.0 + 0.1
            self.__gamma = random.random() * 1000.0 + 0.1
            self.__delta = random.random() * 10.0 + 0.1
            self.__sigma1 = random.random() * 1.0 + 0.1
            self.__sigma2 = random.random() * 1.0 + 0.1
            for j in range(-self.__C + 1, -1):
                for k in range(-self.__C0 + 1, -1):
                    self.__A = j
                    self.__B = k
                    m1, m2, m3, m4 = self.Ms(self.__A, self.__B, self.__sigma1, self.__sigma2)
                    tmp  = self.J(m1, m2, m3, m4)
                    if (max < tmp):
                        max = tmp
                        self.__coefArray[i][0] = self.__alpha
                        self.__coefArray[i][1] = self.__beta
                        self.__coefArray[i][2] = self.__gamma
                        self.__coefArray[i][3] = self.__delta
                        self.__ABArray[i][0] = self.__A
                        self.__ABArray[i][1] = self.__B
            if (-self.__ABArray[i][1] >= 10):
                self.__targetArray[i] = 1
            if (-self.__ABArray[i][1] < 10):
                self.__targetArray[i] = 0
            print( "A ", self.__ABArray[i][0], "B ", self.__ABArray[i][1])
        return self.__coefArray, self.__ABArray, self.__targetArray


# j = LinDataSet(10)
# coefs, abs, target = j.Find()
# print(coefs)
# print(abs)

            

    