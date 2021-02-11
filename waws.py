import os
import configparser
import numpy as np
from scipy import signal
import statsmodels.tsa.api as smt
import matplotlib.pyplot as plt
import numba
import time

@numba.jit(nopython=True)
def harris(v10, I10, alpha, f, z):
    # WARNING: check it again
    # vz = v10 * (z / 10) ** alpha
    u_ = I10 * v10 / np.sqrt(6.677) #WARNING: check it again
    x = f * 1800 / v10
    ret = 4 * (u_ * u_) * x / ((2 + x * x) ** (5.0 / 6)) / f / 2.0
    return ret

@numba.jit(nopython=True)
def davenport(v10, I10, alpha, f, z):
    # WARNING: check it again
    vz = v10 * (z / 10.0) ** alpha
    k = I10 * I10 / 6.0
    x = f * 1200 / v10
    ret = 4.0 * k * v10 * v10 * x * x / ((1 + x * x) ** (4.0 / 3.0)) / f / 2.0
    return ret

@numba.jit(nopython=True)
def simiu(v10, I10, alpha, f, z):
    vz = v10 * (z / 10.0) ** alpha
    u_ = I10 * v10 / np.sqrt(6)
    x = f * z / vz
    ret = np.zeros_like(f)
    for k in range(len(f)):
        if x[k] > 0.2:
            ret[k] = 0.26 * u_ ** 2 / x[k] ** (2 / 3) / f[k] / 2
        else:
            ret[k] = 200 * x[k] * u_ ** 2 / f[k] / ((1 + 50 * x[k]) ** (5 / 3.0)) / 2.0
    return ret

@numba.jit(nopython=True)
def coherence():
    # TODO:
    for i in range(npts):
        for j in range(npts):
            coh[:,i,j] = np.exp(-2 * f * np.sqrt(cx * cx * (x[i] - x[j]) ** 2 + cy * cy * (y[i] -y[j]) ** 2 + cz * cz * (z[i] - z[j]) ** 2) / (vz[i] + vz[j]))

@numba.jit(nopython=True)
def cross_spectrum(Sw, coh):
    npts = Sw.shape[-1]
    for i in range(npts):
        for j in range(npts):
            Sw[:,i,j] = (np.sqrt(Sw[:,i,j] * Sw[:,i,j]) * coh[:,i,j])


@numba.jit(nopython=True)
def synthesis(Hw, nfreq, m, dw, npts, phi, t):
    # npts = Hw.shape[-1]
    vt = np.zeros((m, npts))
    for i in range(npts):
        for j in range(nfreq):
            for k in range(i+1):
                wml = j * dw + (1+k) * dw / npts
                tmp = 2 * np.sqrt(dw / 2.0 / np.pi) * np.cos(wml * t + phi[k, j])
                vt[:,i] += np.abs(Hw[j,k,i]) * tmp
    return vt

@numba.jit(nopython=True)
def fft_synthesis():
    pass

class ConfigData(object):
    def __init__(self, fname):
        self.config = configparser.ConfigParser()
        self.config.read(fname)
        self.parse()

    def _parse_wind(self):
        wind = self.config["wind"]
        self.v10 = wind.getfloat("reference wind speed (m/s)")
        self.I10 = wind.getfloat("reference turbulence intensity")
        self.alpha = wind.getfloat("alpha")
        self.spectrum_type = wind.get("type of wind spectrum").strip().lower()
        self.coh_type = wind.get("type of coherence function").strip().lower()
        self.cx = wind.getfloat("cx")
        self.cy = wind.getfloat("cy")
        self.cz = wind.getfloat("cz")

    def _parse_terrain(self):
        terrain = self.config["terrain"]
        self.karman_const = terrain.getfloat("Karman constant")
        self.z0 = terrain.getfloat("z0")

    def _parse_waws(self):
        waws = self.config["waws"]
        self.total_time = waws.getfloat("total time of simulated wind (s)")
        unit = waws.get("unit of frequency (Hz/Pi)")
        omega_up = waws.getfloat("upper bound of cut-off frequency")
        if unit.lower() == "pi":
            self.omega_up = omega_up * np.pi
        elif unit.lower() == "hz":
            self.omega_up = omega_up
        else:
            raise ValueError("unit of frequency (Hz/Pi)?")
        self.num_freq = waws.getint("number of segments of frequency")

    def _parse_points(self):
        points = self.config["points"]
        self.is_read_points = points.getboolean("read points data or not(yes/no)")

    def _parse_file(self):
        direct = self.config["file"]
        self.workdir = direct.get("working directory")

    def parse(self):
        self._parse_wind()
        self._parse_terrain()
        self._parse_waws()
        self._parse_points()
        self._parse_file()


class GustWindField(object):

    def __init__(self, config):
        if not isinstance(config, ConfigData):
            raise ValueError("config must be an ConfigData object!")
        # parse the parameters from config object
        # wind parameters
        self.v10 = config.v10
        self.I10 = config.I10
        self.alpha = config.alpha
        self.spectrum_type = config.spectrum_type
        self.coherence_type = config.coh_type
        self.cx = config.cx
        self.cy = config.cy
        self.cz = config.cz

        # terrain parameters
        self.k = config.karman_const
        self.z0 = config.z0

        # waws parameters
        self.t = config.total_time
        self.wup = config.omega_up
        self.N = config.num_freq
        self.dw = self.wup / self.N
        self.dt = 10 ** np.floor(np.log10(np.pi / self.wup))   # dt <= pi / wup

        # file parameters
        self.workdir = os.path.abspath(config.workdir)

        # points parameters
        fname = os.path.join(self.workdir, "points.csv")
        if config.is_read_points:
            self.points = np.loadtxt(fname, delimiter=",", skiprows=1)
            # self.npts = len(self.points)
        else:
            self.points = None
            # self.num_points = None
            print("Please assign simulated points")

        # other attributes
        self.coh = None
        self.Sw = None
        self.Hw = None

    def set_points(self, points):
        if not isinstance(points, np.ndarray):
            raise ValueError("points must be ndarray")
        self.points = points
        # self.num_points = len(points)

    def _spectrum(self):
        if self.points is None:
            raise UnboundLocalError("Does not exsit simulated points!")
        npts = len(self.points)
        self.Sw = np.zeros((self.N, npts, npts))

        z = self.points[:,3]
        vz = self.v10 * (z / 10) ** self.alpha
        u_star = self.I10 * self.v10 / np.sqrt(6)
        f = np.arange(self.dw, self.wup + self.dw, self.dw) / 2.0 / np.pi
        command = self.spectrum_type + "(self.v10, self.I10, self.alpha, f, z[i])"
        for i in range(npts):
            self.Sw[:,i,i] = eval(command)

    def _coherence(self):
        npts = len(self.points)
        self.coh = np.zeros((self.N, npts, npts))
        x = self.points[:,1]
        y = self.points[:,2]
        z = self.points[:,3]
        cx, cy, cz = self.cx, self.cy, self.cz
        f = np.arange(self.dw, self.wup + self.dw, self.dw) / 2.0 / np.pi
        z = self.points[:,3]
        vz = self.v10 * (z / 10) ** self.alpha
        

    def _cross_spectrum_matrix(self):
        if self.Sw is None or self.coh is None:
            raise UnboundLocalError("Does not generate spectrum or coherence!")
        npts = len(self.points)
        for i in range(npts):
            for j in range(npts):
                self.Sw[:,i,j] = (np.sqrt(self.Sw[:,i,j] * self.Sw[:,i,j]) *
                                  self.coh[:,i,j])

    def _cholesky(self):
        if self.Sw is None:
            raise UnboundLocalError("Does not generate spectrum martix!")
        self.Hw = np.zeros_like(self.Sw)
        for i in range(self.N):
            self.Hw[i,:,:] = np.linalg.cholesky(self.Sw[i,:,:]) 

    def generate(self, method="direct"):
        """[summary]

        Args:
            method (str, optional): [description]. Defaults to "direct".
        """
        # generate cross sprectrum matrix
        self._spectrum()
        self._coherence()
        self._cross_spectrum_matrix()
        # cholesky decomposion
        self._cholesky()
        t = np.arange(self.dt, self.t + self.dt, self.dt)
        m = len(t)
        npts = len(self.points)
        self.vt = np.zeros((m, npts))
        # generate random phase
        np.random.seed(0)
        phi = 2 * np.pi * np.random.rand(npts, self.N)
        dw = self.dw
        if method == "direct":
            self.vt = synthesis(self.Hw, self.N, m, dw, npts, phi, t)
        elif method == "fft":
            # TODO:
            self.vt = fft_synthesis()
        else:
            raise ValueError("Unrecongnized method: " + method)

    def error(self):
        # plot time history
        t = np.arange(self.dt, self.t + self.dt, self.dt)
        fig,ax = plt.subplots()
        ax.plot(t, self.vt[:,0], lw=1, c="black")
        # ax.plot(t_sieres, vz[0], lw=2, c="red")
        plt.show()
        plt.close(fig)

        # power spectrum
        z = self.points[:,3]
        fs = 1.0 / self.dt
        fxx, pxx = signal.welch(self.vt[:,1], fs=fs, window="hann", 
                                nperseg=4096, scaling="density")
        # remove f > self.wup / 2 / pi
        ind = np.where(fxx < self.wup / 2.0 / np.pi)
        fxx = fxx[ind]
        pxx = pxx[ind]

        fig, ax = plt.subplots()
        ax.loglog(fxx, pxx, c="red", lw=1)
        ax.loglog(fxx[1:], harris(self.v10, self.I10, self.alpha, fxx[1:], z[1]),
                  c="black", lw=1)
        plt.show()
        plt.close(fig)


def foo(x):
    s = 0
    for i in range(x):
        s += i
    return s

@numba.jit(nopython=True)
def fast_foo(x):
    s = 0
    for i in range(x):
        s += i
    return s


if __name__ == "__main__":
    # start = time.time()
    # config = ConfigData("config.ini")
    # gust = GustWindField(config)
    # points = np.zeros((5,4))
    # for i in range(1,6):
    #     points[i-1, 3] = i * 10
    # gust.set_points(points)
    # gust.generate()
    # end = time.time()
    # print("cost: ", end - start)
    # gust.error()

    x = int(1e8)
    start = time.time()
    s = foo(x)
    mid = time.time()
    print("foo costs: ", mid - start)
    s = fast_foo(x)
    end = time.time()
    print("fast_foo costs: ", end - mid)

