from ctypes.wintypes import DOUBLE
from re import S
from ssl import SSL_ERROR_EOF
from wsgiref import headers
import numpy as np
import random
from pandas import DataFrame
from scipy import integrate
from prettytable import PrettyTable
from scipy.optimize import minimize
from matplotlib import pyplot as plt
from tabulate import tabulate as tbl
from scipy import integrate
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

class LinAprox:
    __n = 10
    random.seed(2)
    __C = 140
    __C0 = 60
    __setParams = np.zeros((__n, 6))
    __set = np.zeros((__n, 15))
    __x = np.zeros((__n, 4))
    __y = np.zeros(__n)
    def __init__(self):
        pass

    def __E(self, x, A, B, sigma1):
        return sigma1 * ((A + B * np.cos(2*np.pi*x)) + self.__C)

    def __S(self, x, A, B, sigma2):
        return sigma2 * ((A + B * np.cos(2*np.pi*x)) + self.__C)

    def __Ss(self, x):
        return np.cos(2 * np.pi * x) + 1

    def __G(self, x, A, B):
        return np.power((A + B * np.cos(2*np.pi*x)) + self.__C0, 2)

    def M1(self, x, A, B, sigma1):
        return self.__E(x, A, B, sigma1)

    def M2(self, x, A, B, sigma2):
        return -(self.__Ss(x) * self.__S(x, A, B, sigma2))

    def M3(self, x, A, B):
        return -((A + B * np.cos(2*np.pi*x))*(A + B * np.cos(2*np.pi*x)))

    def M4(self, x, A, B):
        return -(self.__G(x, A, B))

    def MakeParamSet(self):
        for i in range(0, self.__n):
            self.__setParams[i][0] = round(random.random() * 300.0 + 0.1, 2) #alpha
            self.__setParams[i][1] = round(random.random() * 1.0 + 0.1, 2) #beta
            self.__setParams[i][2] = round(random.random() * 300.0 + 0.1, 2) #gamma
            self.__setParams[i][3] = round(random.random() * 10.0 + 0.1, 2) #delta
            self.__setParams[i][4] = round(random.random() * 1.0 + 0.1, 2) #sigma 1
            self.__setParams[i][5] = round(random.random() * 1.0 + 0.1, 2) #sigma 2
        return self.__setParams

    def MakeXSetI(self):
        for i in range(0, self.__n):
            alpha = self.__setParams[i][0]
            beta = self.__setParams[i][1]
            gamma = self.__setParams[i][2]
            delta = self.__setParams[i][3]
            sigma1 = self.__setParams[i][4]
            sigma2 = self.__setParams[i][5]
            self.__set[i][11] = i
            print(i)
            n = 0
            tmp = -99999999999
            for A in range(-self.__C + 1, -1):
                for B in range(-self.__C0 + 1, -1):
                    M1, err = integrate.quad(self.M1, 0, 1, args = (A, B, sigma1))
                    M2, err = integrate.quad(self.M2, 0, 1, args = (A, B, sigma2))
                    M3, err = integrate.quad(self.M3, 0, 1, args = (A, B))
                    M4, err = integrate.quad(self.M4, 0, 1, args = (A, B))
                    J = alpha * M1 + gamma * M2 + beta * M3 + delta * M4
                    if(tmp < J):
                        tmp = J
                        self.__set[i][0] = alpha
                        self.__set[i][1] = beta
                        self.__set[i][2] = gamma
                        self.__set[i][3] = delta
                        self.__set[i][4] = M1
                        self.__set[i][5] = M2
                        self.__set[i][6] = M3
                        self.__set[i][7] = M4
                        self.__set[i][9] = A
                        self.__set[i][10] = B
                        self.__set[i][12] = sigma1
                        self.__set[i][13] = sigma2
                        self.__set[i][14] = J
                        self.__x[i][0] = alpha
                        self.__x[i][1] = beta
                        self.__x[i][2] = gamma
                        self.__x[i][3] = delta
                        
                        
                        if(-B > 9 and -B < 11):
                            self.__set[i][8] = 2
                            self.__y[i] = 2
                        if(-B >= 11):
                            self.__set[i][8] = 1
                            self.__y[i] = 1
                        if(-B <= 9):
                            self.__set[i][8] = 0
                            self.__y[i] = 0
        return self.__set, self.__x, self.__y

    
    def Net(self):
        # if (self.__y[1] == 1):
        #     xtt = np.zeros(11)
        #     t = 0
        #     for i in range(0, 11):
        #         xtt[i] = self.__set[1][9] + self.__set[1][10]*np.cos(2*np.pi*t)
        #         t += 0.1
        #     t = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        x, xt, y, yt = train_test_split(self.__x, self.__y,
         test_size=0.3, random_state=42, stratify=self.__y)

        ppnrl = MLPClassifier(hidden_layer_sizes=(80, 60, 20 ), activation="relu", solver="lbfgs",
         random_state=42, max_iter=100, learning_rate="constant").fit(x, y)
        print(ppnrl.score(x, y))

        plt.hist(y)
        plt.title("Targets")
        plt.show()

        train_sizesrl, train_scoresrl, test_scoresrl = learning_curve(estimator=ppnrl, X=x, y=y,cv=10, train_sizes=np.linspace(0.1, 1.0, 10))

        train_meanrl = np.mean(train_scoresrl, axis=1)
        train_stdrl = np.std(train_scoresrl, axis=1)
        test_meanrl = np.mean(test_scoresrl, axis=1)
        test_stdrl = np.std(test_scoresrl, axis=1)

        plt.figure(1)
        plt.plot(train_sizesrl, train_meanrl, color='blue', marker='o', markersize=5, label='Training Accuracy')
        plt.fill_between(train_sizesrl, train_meanrl + train_stdrl, train_meanrl - train_stdrl, alpha=0.15, color='blue')
        plt.plot(train_sizesrl, test_meanrl, color='green', marker='+', markersize=5, linestyle='--', label='Validation Accuracy')
        plt.fill_between(train_sizesrl, test_meanrl + test_stdrl, test_meanrl - test_stdrl, alpha=0.15, color='green')
        plt.title('Learning Curve')
        plt.xlabel('Training Data Size')
        plt.ylabel('Model accuracy')
        plt.grid()
        plt.legend(loc='lower right')
        plt.show()

        

if __name__ == "__main__":
    j = LinAprox()
    j.MakeParamSet()
    data, x, y = j.MakeXSetI()
    #j.Net()

    print(tbl(data, headers=["alpha", "beta", "gamma", "delta", "M1", "M2", "M3", "M4", "Target", "A", "B", "i", "sigma1", "sigma2", "J"], tablefmt="fancy_grid"))