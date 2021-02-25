import os
import waws
import numpy as np
import scipy
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
from scipy.integrate import quad

if __name__ == "__main__":
    # fname = "config.ini"
    # config = waws.ConfigData(fname)
    # gust = waws.GustWindField(config)
    # gust.generate()
    # gust.error()

    # generate a random signal
    # np.random.seed(0)
    # n = 1000
    # x = np.random.rand(n)
    # t = np.linspace(0, 60, 1000)
    # dt = t[1] - t[0]
    # fs = 1 / dt

    # # spectrum to auto-correlation
    # fxx, pxx = scipy.signal.welch(x, fs=fs, window="hann", nperseg=512,
    #            scaling="density", return_onesided=True)
    # Rxx = np.fft.ifft(pxx, n=n)
    # Rxx /= np.max(Rxx)
    # # t = np.fft.
    # t1 = np.arange(n) * dt

    # # auto-correlation 
    # # xcorr = smt.stattools.acf(x, nlags=n-1, fft=False) * x.var()
    # xcorr = np.correlate(x, x, mode="full")
    # xcorr /= np.max(xcorr)

    # fig, axs = plt.subplots(nrows=2, ncols=2)
    # axs = axs.flatten()
    # axs[0].plot(t, x)
    # axs[1].loglog(fxx, pxx, c="red", lw=1)
    # axs[2].plot(t1, Rxx.imag, c="black", lw=1)
    # # axs[2].acorr(x, maxlags=n-20)
    # axs[2].plot(t1, xcorr[len(xcorr)//2:], c="red", lw=1, ls="dashed")
    # plt.show()
    # plt.close(fig)

    Iu = 0.14
    v10 = 30
    print("simga2: ", (Iu * v10) ** 2)
    print(quad(waws.davenport, 0, np.inf, args=(v10, Iu)))