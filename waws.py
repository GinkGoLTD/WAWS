import os
import time
import configparser
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import numba
np.seterr(divide="print")


###############################################################################
#                              Spectra Function                               #
###############################################################################
def davenport(f, sigma2, **kwargs):
    """ one-side Davenport spectrum, Davenport 1961, adopted in Chinese code

    Args:
        f (1d-ndarray): freqency, unit: Hz
        sigma2 (float): variance, (Iu * vz) ** 2

        **kwargs: spectrum propeteries
        Lu (float): integral scale, unit:m
        v10 (float): mean wind speed at 10m height, unit: m/s

    Returns:
        1d-ndarray: specturm value
    """
    if "Lu" in kwargs:
        Lu = kwargs["Lu"]
    else:
        Lu = 1200.0
    v10 = kwargs["v10"]
    x = f * Lu / v10
    ret = sigma2 * 2.0 * x * x / ((1 + x * x) ** (4.0 / 3.0)) / f / 3.0
    return ret

def karman(f, sigma2, **kwargs):
    """ one-side Von Karman spectrum 
    
    Args:
        f (1d-ndarray): freqency, unit: Hz
        sigma2 (float): variance, (Iu * vz) ** 2

        **kwargs: spectrum propeteries
        z (float): height, unit:m
        vz (float): mean wind speed at z height, unit: m/s
    """


    vz = kwargs["vz"]
    z = kwargs["z"]
    Lu = 100 * (z / 30) ** 0.5
    x = f * Lu / vz
    ret = sigma2 / f * 4.0 * x / ((1 + 70.8 * x * x) ** (5.0 / 6))
    return ret


def harris(f, sigma2, **kwargs):
    """ one-side Harris spectra, adopted in Austrilia code

    Args:
        f (1d-ndarray): freqency, unit: Hz
        sigma2 (float): variance, (Iu * vz) ** 2

        **kwargs: spectrum propeteries
        v10 (float): mean wind speed at 10m height, unit: m/s
    """
    v10 = kwargs["v10"]
    x = f * 1800 / v10
    ret = sigma2 / f * 0.6 * x / ((2 + x * x) ** (5.0 / 6))
    return ret

def simiu(f, sigma2, **kwargs):
    """ one-side Simiu spectra

    Args:
        f (1d-ndarray): freqency, unit: Hz
        sigma2 (float): variance, (Iu * vz) ** 2

        **kwargs: spectrum propeteries
        z (float): height, unit:m
        vz (float): mean wind speed at z height, unit: m/s
    """
    z = kwargs["z"]
    vz = kwargs["vz"]
    x = f * z / vz
    ret = np.zeros_like(f, dtype=np.float64)
    for k in range(len(f)):
        if x[k] > 0.2:
            ret[k] = sigma2 / f[k] * 0.0433 / x[k] ** (2.0 / 3)
        else:
            ret[k] = (sigma2 / f[k] * 100.0 * x[k] 
                      / 3.0 / ((1 + 50.0 * x[k]) ** (5 / 3.0)))
    return ret

def kaimal(f, sigma2, **kwargs):
    """ one-side Kaimal spectra, adopted in Amercian code ASCE7

    Args:
        f (1d-ndarray): freqency, unit: Hz
        sigma2 (float): variance, (Iu * vz) ** 2

        **kwargs: spectrum propeteries
        z (float): height, unit:m
        vz (float): mean wind speed at z height, unit: m/s
        l (float)
        epsilon (float)
    """
    vz = kwargs["vz"]
    z = kwargs["z"]
    l = kwargs["l"]
    epsilon = kwargs["epsilon"]

    Lu = l * (z / 10) ** epsilon
    x = f * Lu / vz 
    ret = sigma2 / f * 6.868 * x / ((1 + 10.302 * x) ** (5.0 / 3))
    return ret


###############################################################################
#                   Helper Function for GustWindField class                   #
###############################################################################
@numba.jit(nopython=True)
def cross_spectrum(Sw, coh):
    npts = Sw.shape[-1]
    for i in range(npts):
        for j in range(npts):
            Sw[:,i,j] = (np.sqrt(Sw[:,i,j] * Sw[:,i,j]) * coh[:,i,j])


@numba.jit(nopython=True)
def synthesis(Hw, nfreq, m, dw, npts, phi, t):
    # npts = Hw.shape[-1]
    vt = np.zeros((m, npts), dtype=np.float64)
    for i in range(npts):
        for j in range(nfreq):
            for k in range(i+1):
                wml = j * dw + (1+k) * dw / npts
                tmp = 2.0 * np.sqrt(dw) * np.cos(wml * t + phi[k,j])
                vt[:,i] += np.abs(Hw[j,i,k]) * tmp
    return vt

def fft_synthesis(Hw, nfreq, m, dw, npts, phi, t):
    vt = np.zeros((m, npts), dtype=np.float64)
    B = np.zeros((nfreq, npts, npts), dtype=complex)
    phi_ml = np.zeros((nfreq, npts, npts), dtype=complex)

    for i in range(npts):
        for j in range(npts):
            phi_ml[:,i,j] = np.exp(1.0j * phi[j,:])

    B = 2.0 * np.sqrt(dw) * phi_ml * Hw
    Gjm = m * np.fft.ifft(B, n=m, axis=0)

    tmp = np.zeros_like(Gjm)
    for i in range(npts):
        for j in range(npts):
            tmp[:,i,j] = np.exp(1.0j * dw * t * (j + 1.0) / (npts * 1.0))

    tmp = Gjm * tmp
    for j in range(npts):
        for m in range(j+1):
            vt[:,j] += tmp[:,j,m].real

    return vt


###############################################################################
#                           data visualization function                       #
###############################################################################
def plot_time_history(t, x, pid, path=None):
    fig,ax = plt.subplots(figsize=np.array([12,6])/2.54, tight_layout=True)
    ax.plot(t, x, lw=0.5, c="black")
    ax.axhline(np.mean(x), c="red", lw=2, ls="dashed")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("wind speed (m/s)")
    ax.grid(True)
    plt.title("wind velocity of point " + str(pid))
    if path is None:
        plt.pause(5)
    else:
        figname = os.path.join(path, 
                  "wind_speed_of_point_" + str(pid) + ".svg")
        plt.savefig(figname, dpi=300)
    plt.close(fig)
    

def plot_spectrum(f, sf, t, x, pid=None, path=None):
    # power spectrum, S(f)
    fs = 1.0 / (t[1] - t[0])
    fxx, pxx = signal.welch(x.flatten(), fs=fs, window="hann", 
                            nperseg=len(t), scaling="density")
    # remove f > self.wup / 2 / pi
    ind = np.where(fxx < f[-1])
    fxx = fxx[ind][1:]
    pxx = pxx[ind][1:]

    fig, ax = plt.subplots(figsize=np.array([8,6])/2.54, tight_layout=True)
    ax.loglog(fxx, pxx, c="black", lw=0.5, label="simulated")
    ax.loglog(f, sf, c="red", lw=2, label="target")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel(r"$S(f) (m^2/s)$")
    ax.legend()
    ax.grid(True)
    if path is None:
        plt.pause(5)
    else:
        figname = os.path.join(path, "spectrum_of_point_" + str(pid) + ".svg")
        plt.savefig(figname, dpi=300)
    plt.close(fig)

def plot_coherence(f, cxy, f_, cxy_, figname=None):
    fig, ax = plt.subplots(figsize=np.array([8,7])/2.54, tight_layout=True)
    ax.semilogx(f_, cxy_, c="black", lw=1, label="simulated")
    ax.semilogx(f, cxy, c="red", lw=2, label="target")
    ax.grid(True)
    ax.set_xlabel("f (Hz)", fontsize=12, fontstyle="italic")
    ax.set_ylabel("coherence function", fontsize=12)
    plt.legend()
    if figname is None:
        plt.pause(5)
    else:
        plt.savefig(figname, dpi=300)
    plt.close(fig)

def plot_stats(Xz, z, Xz_, z_, ylabel,figname=None):
    fig, ax = plt.subplots(figsize=np.array([8,7])/2.54, tight_layout=True)
    ax.plot(Xz, z, c="red", lw=2, label="target")
    ax.scatter(Xz_, z_, c="black", s=20, marker="o", label="simulated", 
               zorder=2.5)
    ax.grid(True)
    ax.legend(loc="best")
    ax.set_xlabel(ylabel)
    ax.set_ylabel("z (m)")
    if figname is None:
        plt.pause(5)
    else:
        plt.savefig(figname, dpi=300)
    plt.close(fig)


###############################################################################
#                               ConfigData class                              #
###############################################################################
class ConfigData(object):
    def __init__(self, fname):
        self.config = configparser.ConfigParser()
        self.config.read(fname)
        self.parse()

    def _parse_wind(self):
        wind = self.config["wind"]
        self.v10 = wind.getfloat("reference wind speed (m/s)")
        self.alpha = wind.getfloat("alpha")
        self.I10 = wind.getfloat("reference turbulence intensity")
        self.d = wind.getfloat("d")
        self.spectrum_type = wind.get("type of wind spectrum").strip().lower()
        if self.spectrum_type.lower() == "kaimal":
            self.l = wind.getfloat("l (m)")
            self.epsilon = wind.getfloat("epsilon")
        else:
            self.l, self.epsilon = None, None
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
        self.num_time = waws.getint("number of segments of time")
        self.double_index = waws.getboolean(
            "double indexing frequency (yes/no)")

    def _parse_file(self):
        files = self.config["file"]
        path = files.get("working directory")
        if os.path.isdir(path):
            self.workdir = os.path.abspath(path)
        else:
            self.workdir = os.path.abspath("./")
        self.is_read_points = files.getboolean("read points data (yes/no)")
        compare_points = files.get("comparing points ID")
        self.compare_points = eval(compare_points)

    def parse(self):
        self._parse_wind()
        self._parse_terrain()
        self._parse_waws()
        self._parse_file()


###############################################################################
#                            GustWindField class                              #
###############################################################################
class GustWindField(object):
    """ Gust wind field simulation
        1) wind profile is governed by power law vz = v10 * (z / 10) ** alpha
        2) support different type wind spectrum
        3) only support coherence function proposed by Davenport

    Args:
        workdir (float): working directory
        points (2d-ndarray): simulated points, [[id, x, y, z], ...]
        v10 (float): reference wind speed at 10m height
        I10 (float): reference turbulence intensity at 10m height
        alpha (float): wind profile power exponent
        spectrum (string): wind spectrum type
        coherence (string): coherence function type, not support yet
        cx, cy, cz (float): coefficients of coherence function
        k (float): karman constant
        z0 (float): ground roughness length
        t (float): total time of the simulated wind history
        wup (float): upper boundary of the cut-off frequency, (rad/s)
        N (int): number of frequncy segments
        M (int): number of time segments
        dw (float): frequency step, wup / N
        dt (float): time step

        Sw (3d-ndarray): cross spectrum matrix
        Hw (3d-ndarray): Hw = cholesky(Sw)
        coh (3d-ndarray): coherence coeffiecient matrix

    Methods:
        set_points(points): self.points = points
        generate(): generate the gust wind field
        error(): error analysis
    """
    def __init__(self, config, points=None):
        if not isinstance(config, ConfigData):
            raise ValueError("config must be an ConfigData object!")

        # parse the parameters from config object
        # file parameters
        self.workdir = os.path.abspath(config.workdir)
        # create results file
        path = os.path.join(self.workdir, "results")
        if not os.path.exists(path):
            os.makedirs(path)
        # load points
        fname = os.path.join(self.workdir, "points.csv")
        if config.is_read_points:
            self.points = np.loadtxt(fname, delimiter=",", skiprows=1)
        else:
            if not isinstance(points, np.ndarray):
                raise ValueError("points must be ndarray")
            self.points = points
        self.target_PIDs = config.compare_points

        # wind parameters
        self.v10 = config.v10
        self.alpha = config.alpha
        self.I10 = config.I10
        self.d = config.d
        self.spectrum = config.spectrum_type.lower()
        self.l, self.epsilon = config.l, config.epsilon
        self.coherence = config.coh_type.lower()
        self.cx = config.cx
        self.cy = config.cy
        self.cz = config.cz
        self.vz = self.v10 * (self.points[:,3] / 10) ** self.alpha
        self.Iu = self.I10 * (self.points[:,3] / 10) ** (-self.d)

        # terrain parameters
        self.k = config.karman_const
        self.z0 = config.z0

        # waws parameters
        self.T = config.total_time
        self.wup = config.omega_up
        self.N = config.num_freq
        self.M = config.num_time
        self.double_index = config.double_index
        self._waws_parameters()

        # other attributes
        self.coh = None
        self.Sw = None
        self.Hw = None

    @staticmethod
    def wind_profile(z, v10, alpha):
        return v10 * (z / 10) ** alpha

    @staticmethod
    def turbulence_intensity(z, c, d):
        return c * (z / 10) ** (-d)

    def _waws_parameters(self):
        """ set wml and t """
        # check M/N
        if (self.M < 2 * self.N):
            raise Warning("M/N < 2!")

        # set dw and wml
        self.dw = self.wup / self.N
        if not self.double_index:
            self.wml = np.arange(self.dw, self.wup+self.dw, self.dw)
        else:
            npts = len(self.points)
            w = [i * self.dw for i in range(self.N)]
            self.wml = [] 
            for i in range(1, npts+1):
                self.wml += [x + i / npts * self.dw for x in w]
            self.wml = np.array(self.wml)

        # set dt and t
        self.dt = 2 * np.pi / self.M / self.dw
        self.t = np.arange(self.M) * self.dt
        if (self.T > 2.0 * np.pi / self.dw):
            raise Warning("T > T0! Please increase N!")

    def _spectrum(self, func):
        if self.points is None:
            raise UnboundLocalError("Does not exsit simulated points!")
        npts = len(self.points)
        self.Sw = np.zeros((len(self.wml), npts, npts), dtype=np.float64)

        for i in range(npts):
            sigma2 = (self.Iu[i] * self.vz[i]) ** 2
            kwargs = {"v10": self.v10, "vz": self.vz[i], "z": self.points[i,3],
                      "l": self.l, "epsilon": self.epsilon}
            self.Sw[:,i,i] = (func(self.wml / 2.0 / np.pi, sigma2, **kwargs) 
                              / 2.0 / np.pi / 2.0)

        if not self.double_index:
            self.target_Sw = self.Sw   # single indexing frequency
        else:
            self.target_Sw = self.Sw[:self.N,:,:]  # double indexing frequency

    def _coherence(self):
        npts = len(self.points)
        self.coh = np.zeros((len(self.wml), npts, npts), dtype=np.float64)
        x = self.points[:,1]
        y = self.points[:,2]
        z = self.points[:,3]
        cx, cy, cz = self.cx, self.cy, self.cz

        for i in range(npts):
            for j in range(npts):
                self.coh[:,i,j] = np.exp(-2.0 * self.wml / 2 / np.pi * 
                                  np.sqrt(cx * cx * (x[i] - x[j]) ** 2 +
                                          cy * cy * (y[i] - y[j]) ** 2 +
                                          cz * cz * (z[i] - z[j]) ** 2) /
                                  (self.vz[i] + self.vz[j]))

    def _cross_spectrum_matrix(self):
        if self.Sw is None or self.coh is None:
            raise UnboundLocalError("Does not generate spectrum or coherence!")
        npts = len(self.points)
        for i in range(npts):
            for j in range(npts):
                self.Sw[:,i,j] = (np.sqrt(self.Sw[:,i,i] * self.Sw[:,j,j]) *
                                  self.coh[:,i,j])

    def _cholesky(self):
        if self.Sw is None:
            raise UnboundLocalError("Does not generate spectrum martix!")
        npts = len(self.points)
        Hw = np.zeros_like(self.Sw,)
        for i in range(len(self.wml)):
            Hw[i,:,:] = np.linalg.cholesky(self.Sw[i,:,:])

        self.Hw = np.zeros((self.N, npts, npts))
        if not self.double_index:
            self.Hw = Hw
        else:
            for i in range(npts):
                s, e = i * self.N, (i + 1) * self.N
                self.Hw[:,:,i] = Hw[s:e,:,i]

    def generate(self, mean=True, method="fft"):
        """
        Args:
            mean (bool): Including mean wind speed or not? Defaults to True
            method (str, optional): ["fft" or "Deodatis"]. Defaults to "direct".
        """
        method = method.lower()
        if method not in ["fft", "deodatis"]:
            raise ValueError("unrecongnized method! fft or deodatis?")

        print("generate cross spectrum matrix...")
        self._spectrum(eval(self.spectrum))
        self._coherence()
        self._cross_spectrum_matrix()

        print("cholesky decompostion...")
        self._cholesky()
        npts = len(self.points)
        self.vt = np.zeros((self.M, npts), dtype=np.float64)

        print("synthesis gust wind speed...")
        np.random.seed(0)
        phi = 2 * np.pi * np.random.rand(npts, self.N)
        dw = self.dw
        if method == "deodatis":
            self.vt = synthesis(self.Hw, self.N, self.M, dw, npts, phi, self.t)
        elif method == "fft":
            self.vt = fft_synthesis(self.Hw, self.N, self.M, dw, npts, phi,
                                    self.t)
        else:
            raise ValueError("Unrecongnized method: " + method)
        self.vt -= np.mean(self.vt, axis=0)
        if mean:
            self.vt += self.vz

        print("finished!")

    def save(self):
        # create directory
        path = os.path.join(self.workdir, "results")
        if not os.path.exists(path):
            os.makedirs(path)

        # save wind speed
        fname = os.path.join(path, "wind_speed.csv")
        ans = np.hstack((self.t.reshape(-1,1), self.vt))
        head = ["t"] + [str(self.points[i,0]) for i in range(len(self.points))]
        np.savetxt(fname, ans, delimiter=",", header=",".join(head))

        # save traget spectrums
        fname = os.path.join(path, "target_spectrum.csv")
        npts = len(self.points)
        target = np.zeros((self.N, npts), dtype=np.float64)
        for i in range(npts):
            for j in range(npts):
                target[:,i] = self.target_Sw[:,i,j] * 2.0 * np.pi
        if not self.double_index:
            freq = self.wml
        else:
            freq = self.wml[:self.N]
        freq = freq / 2.0 / np.pi
        ans = np.hstack((freq.reshape(-1,1), target))
        head = ["f(Hz)"] + [str(self.points[i,0]) for i in range(npts)]
        np.savetxt(fname, ans, delimiter=",", header=",".join(head))

    def stats_test(self, save=True):
        # check wind profile
        z = np.arange(0, np.max(self.points[:,3])+1, 1)
        vz = self.v10 * (z / 10) ** self.alpha
        z_ = self.points[:,3].reshape(-1,1)
        vz_ = np.mean(self.vt, axis=0).reshape(-1,1)
        # plot
        figname = os.path.join(self.workdir, "results", "mean.svg")
        plot_stats(vz, z, vz_, z_, "V (m/s)", figname)

        # check turbulence intensity
        z = np.arange(0, np.max(self.points[:,3]), 1)
        Iz = GustWindField.turbulence_intensity(z, self.I10, self.d)
        Iz_ = np.std(self.vt, axis=0) / self.vz
        Iz_ = Iz_.reshape(-1,1)
        # plot
        figname = os.path.join(self.workdir, "results", "turbulence.svg")
        plot_stats(Iz, z, Iz_, z_, "Iu (m/s)", figname)

        # save all the data
        data = np.hstack((z_, vz_, Iz_))
        fname = os.path.join(self.workdir, "results", "vz_Iu.csv")
        np.savetxt(fname, data, delimiter=",", header="z(m), vz(m/s), Iu")

    def coherence_test(self):
        f = self.wml[1:self.N] / 2.0 / np.pi
        n = len(self.target_PIDs)
        for i in range(n):
            for j in range(i):
                p1, p2 = self.target_PIDs[i], self.target_PIDs[j]
                ind1 = np.where(self.points[:,0]==p1)[0]
                ind2 = np.where(self.points[:,0]==p2)[0]
                cxy = self.coh[1:self.N,ind1,ind2]
                f_, cxy_ = signal.coherence(self.vt[:,ind1].flatten(), 
                                self.vt[:,ind2].flatten(), fs=1.0/self.dt,
                                window="hann",nperseg=self.N/4)
                figname = os.path.join(self.workdir, "results", 
                            "coh_p" + str(p1) + "_p" + str(p2) + ".svg")
                plot_coherence(f, cxy, f_, cxy_, figname)


    def _temporal_corr(self, p1, p2, lags):
        ind1 = np.where(self.points[:,0]==p1)[0]
        ind2 = np.where(self.points[:,0]==p2)[0]
        print(ind1)
        Rjk = np.zeros(lags)
        t = np.arange(lags) * self.dt
        w_t = np.zeros((self.N, t.size))
        for i in range(self.N):
            w_t[i,:] = np.cos(self.wml[i] * t.reshape(1,-1))
        print(w_t)
        print(w_t.shape)
        Hw2 = np.zeros((self.N,1))
        for i in range(npts):
            Hw2 += np.abs(self.Hw[:self.N,ind1,i] * 
                   self.Hw[:self.N,ind2,i])

        for i in range(lags):
            tmp = np.sum(Hw2 * w_t[:,i].reshape(-1,1), axis=0) * self.dw * 2
            Rjk[i] = tmp
        return t, Rjk

    def ergodicity_test(self):
        pass

    def error(self):
        # create directory
        path = os.path.join(self.workdir, "results")
        if not os.path.exists(path):
            os.makedirs(path)

        # basic compare, mean wind speed and turbulence intensity
        # self.stats_test()

        # spectrum compare
        for pid in self.target_PIDs:
            # check pid exists?
            ind = np.where(self.points[:,0]==pid)[0]
            if (ind.size == 0):
                raise Warning("Unvalid points ID: ", pid)
            '''
            # plot time_history
            plot_time_history(self.t, self.vt[:,ind], pid, path)

            # compare spectrum
            if not self.double_index:
                freq = self.wml
            else:
                freq = self.wml[:self.N]
            freq = freq / 2.0 / np.pi
            # S(w) = S(f) / 2 / np.pi
            plot_spectrum(freq, 2.0*np.pi*self.target_Sw[:,ind,ind], self.t,
                 self.vt[:,ind], pid, path)
            '''
            # auto-correlation analysis
            R0 = np.fft.ifft(self.target_Sw[:,ind, ind], n=self.M, axis=0)
            t1, R_jk = self._temporal_corr(pid, pid, 1000)
            # Rt = acf(self.vt[:,ind], nlags=self.M, fft=False)
            n = 2000
            Rt = np.zeros(n)
            n = self.M
            Rt = signal.correlate(self.vt[:,ind].flatten(), self.vt[:,ind].flatten(), "same", method="fft")
            print(self.vt.shape)
            Rt /= self.M
            print(Rt.shape)
            lags = 2000
            # Rt1 = autocorrelation(self.vt[:,ind].flatten(), lags)
            # print(Rt1.shape)

            print(np.sum(self.vt[:,ind] * self.vt[:,ind]) / self.M)


            fig, ax = plt.subplots()
            t = np.arange(len(R0)) * self.dt
            ax.plot(t, R0.real, c='black', lw=1)
            ax.plot(t, R0.imag, c='red', lw=1, ls="dashed")
            # ax.plot(np.arange(len(Rt)//2) * self.dt, Rt[len(Rt)//2:], c="green", lw=1, ls="dashed")
            # ax.plot(np.arange(lags) * self.dt, Rt1, c="blue", lw=2)
            # ax.plot(t1, R_jk, c="green")
            plt.show()
            plt.close(fig)

        # # cross-correlation analysis
        # n = len(self.target_PIDs)
        # for i in range(1,n):
        #     for j in range(i):
        #         # if i == j: continue
        #         ind1 = np.where(self.points[:,0]==self.target_PIDs[i])[0]
        #         ind2 = np.where(self.points[:,0]==self.target_PIDs[j])[0]
        #         R0_jk = np.fft.ifft(self.Sw[:self.N,ind1,ind2] / 2 / np.pi, axis=0)
        #         print(R0_jk.shape)
        #         t = np.arange(len(R0_jk)) * self.dt

        #         R_jk = signal.correlate(self.vt[:,ind1].flatten(), self.vt[:,ind2].flatten(), "same") / self.M

        #         fig, ax = plt.subplots()
        #         ax.plot(t, R0_jk.real, c="black", lw=1)
        #         ax.plot(t, R0_jk.imag, c="red", lw=1)
        #         # ax.plot(np.arange(len(R_jk)//2) * self.dt, R_jk[len(R_jk)//2:])
        #         plt.show()
        #         plt.close(fig)

        # coherence analysis



if __name__ == "__main__":
    start = time.time()
    config = ConfigData("config.ini")

    npts = 5
    points = np.zeros((npts,4))
    for i in range(1,npts+1):
        points[i-1, 0] = i
        points[i-1, 3] = i * 10
    gust1 = GustWindField(config, points)
    # gust2 = GustWindField(config, points)
    gust1.generate(mean=True, method="fft")
    # gust1.generate(mean=False, method="deodatis")
    start = time.time()
    # gust1.coherence_test()
    # gust2.generate(method="deodatis")
    # gust1.save()
    end = time.time()
    print("cost: ", end - start)
    gust1.error()

   
