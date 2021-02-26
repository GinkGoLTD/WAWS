import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numba
import multiprocessing as mp

class Father(object):
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    @staticmethod
    def change_age(newage):
        return newage
    
    def change(self, newage):
        self.age = Father.change_age(newage)

def func(name, **kw):
    print("name: ", name)
    print(kw)

def high_func(func, name, **kw):
    print(func)
    print(kw)
    func(name, **kw)

def davenport1(w, v10, I10, sigma2, k):
    f = w / 2.0 / np.pi
    x = f * 1200.0 / v10
    # sigma2 = 6.0 * k * v10 * v10
    # sigma2 = (I10 * v10) ** 2
    return sigma2 * 2.0 / 3.0 * x * x / f / ((1 + x * x) ** (4.0 / 3)) / 2.0 / np.pi

def davenport2(f, v10, I10, simga2, k):
    x = f * 1200.0 / v10
    return simga2 * 2.0 / 3.0 * x * x / f / ((1 + x * x) ** (4.0 /3))

def autocorrelation(x,lags):
    #计算lags阶以内的自相关系数，返回lags个值，分别计算序列均值，标准差
    n = len(x)
    x = np.array(x)
    result = [np.correlate(x[i:]-x[i:].mean(),x[:n-i]-x[:n-i].mean())[0]\
    	/(x[i:].std()*x[:n-i].std()*(n-i)) \
    	for i in range(1,lags+1)]
    return np.array(result)


def coherence(w, cx, cz, dx, dz, Uz1, Uz2):
    return np.exp(-w / 2.0 / np.pi * cz * dz / 0.5 / (Uz1 + Uz2))

import time





if __name__ == "__main__":

    # Sw = np.arange(1, 100*100*200 + 1, 1).reshape(200, 100, 100)
    Sw = np.random.randn(200, 100, 100)
    for i in range(200):
        Sw[i,:,:] = np.matmul(Sw[i,:,:], Sw[i,:,:].transpose()) + np.eye(100) * 0.01
    # Sw = np.ones((200, 100, 100))
    t1 = time.time()
    Hw1 = np.linalg.cholesky(Sw)
    t2 = time.time()
    Hw2 = cholesky(Sw)
    t3 = time.time()
    print("np.linalg.cholesky cost: ", t2 - t1)
    print("cholesky cost: ", t3 - t2)
    print("Hw1:", Hw1[0,:,:])
    print("Hw2:", Hw2[0,:,:])