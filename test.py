import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    # me = Father("ta", 30)
    # print(me.age)
    # me.change(20)
    # print(me.age)
    # func("my")
    # high_func(func,"my", a=1, b=2)
    # high_func(eval("func"), "my")


    # v10 = 30
    # I10 = 0.14
    # d = 0.15
    # z = np.arange(1, 101, 10, dtype=np.float64)
    # Iz = I10 * (z / 10.0) ** (-d)
    # sigma_z = (Iz * z) ** 2

    # k = 0.005
    # simga_z1_ = np.zeros_like(z, dtype=np.float64)
    # simga_z2_ = np.zeros_like(z, dtype=np.float64)

    # for i in range(z.size):
    #     x = integrate.quad(davenport1, 0, np.inf, args=(v10, I10, sigma_z[i], k))
    #     # print(x)
    #     # simga_z1_[i] = float(x[0]) / 2.0 / np.pi
    #     simga_z1_[i] = float(x[0])
    #     # print(type(simga_z_[i]))
    #     # print(simga_z_[i])

    #     y = integrate.quad(davenport2, 0, np.inf, args=(v10, I10, sigma_z[i], k))
    #     simga_z2_[i] = float(y[0])
    
    # # print("sigma2: ", (I10 * v10) ** 2 * 2.0 * np.pi)
    # print(z)
    # print(sigma_z)
    # print(simga_z1_)
    # print(simga_z2_)

    # fig, ax = plt.subplots()
    # ax.scatter(sigma_z, z, marker="s", c="black", s=20)
    # ax.scatter(simga_z1_, z, marker="o", c="red", s=20)
    # ax.scatter(simga_z2_, z, marker="o", c="green", s=20)
    # plt.show()
    # plt.close(fig)

    # f = np.arange(0, 4, 0.1)
    # fig, ax = plt.subplots()
    # ax.plot(f, davenport1(f, v10, I10, sigma_z[1], k), lw=1, c="black")
    # ax.plot(f, davenport2(f, v10, I10, sigma_z[1], k), lw=1, c="red")
    # ax.plot(f, davenport2(f, v10, I10, sigma_z[1], k) / 2.0 / np.pi, c="green")
    # plt.show()
    # plt.close(fig)

    w = np.arange(0, 100, 0.01)
    cx, cz = 7, 8
    dx, dz = 5, 5
    Uz1 = 30
    Uz2 = 32
    coh = coherence(w, cx, cz, dx, dz, Uz1, Uz2)
    cohf = np.exp(-w * cz * dz / 0.5 / (Uz1 + Uz2))
    fig, ax = plt.subplots()
    ax.semilogx(w/2.0/np.pi, coh , c="black", lw=2)
    ax.semilogx(w, cohf, c="red", lw=2, ls="dashed")
    plt.show()
    plt.close(fig)