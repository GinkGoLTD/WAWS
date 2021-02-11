import os
import waws
import numpy as np
import scipy
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # fname = "config.ini"
    # config = waws.ConfigData(fname)
    # gust = waws.GustWindField(config)
    # gust.generate()
    # gust.error()

    np.random.seed(0)
    n = 1000
    x = np.random.rand(n)
    t = np.linspace(0, 60, 1000)
    dt = t[1] - t[0]
    fs = 1 / dt

    # fft to spectrum
    sxx = np.fft.fft(x)
    fx = np.fft.fftfreq(n=t.shape[-1], d=dt)
    sxx = np.abs(sxx[1:n//2]) ** 2 / n / fs
    fx = fx[1:n//2]

    # welch
    fwelch, pwelch = scipy.signal.welch(x, fs=fs, window="hann", nperseg=1024, scaling="density")

    # correlation 
    xcorr = smt.stattools.acf(x, nlags=n-1, fft=False)
    fc = np.fft.fftfreq(n, d=dt)
    fc = fc[1:n//2]
    pc = np.fft.fft(xcorr)
    pc = np.abs(pc)[1:n//2]

    print(fc)
    fig, ax = plt.subplots()
    # ax.plot(t, x, c="black")
    ax.loglog(fx, sxx, c="black", lw=1)
    ax.loglog(fwelch, pwelch, c="red", lw=1)
    # ax.loglog(fc, pc / 2.0, c="green", lw=1)
    plt.show()
    plt.close(fig)
