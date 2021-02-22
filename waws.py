import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import numba
import time
import configparser


###############################################################################
#                              Spectra Function                               #
###############################################################################
@numba.jit(nopython=True)
# def harris(v10, I10, alpha, f, z):
def harris(v10, I10, alpha, w, z, omega=True):
    """ <风荷载规范中若干问题的研究> 夏瑞光
    adopted in Austrilia code
            s(n) = 4 * sigma**2 * x / 6.677 / n / (2 + x**2)**(5/6)
    where, sigma is the turbulence intesisty, RETHINK: at 10m height or not?
           x = 1800 * n / v10;
           x = n * Lu / vz (Austrilia code), Lu = 1000 * (z / 10)**0.25;

    Args:
        v10 ([type]): [description]
        I10 ([type]): [description]
        alpha ([type]): [description]
        f ([type]): [description]
        z ([type]): [description]

    Returns:
        [type]: [description]
    """
    # WARNING: check it again
    # vz = v10 * (z / 10) ** alpha
    # w = 2 * np.pi * f
    u_ = I10 * v10 / np.sqrt(6.677) #WARNING: check it again
    if omega:
        x = w * 1800 / v10 / 2 / np.pi
        ret = 4 * (u_ * u_) * x / ((2 + x * x) ** (5.0 / 6)) / w / 2.0 * 2.0 * np.pi
    else:
        x = w * 1800 / v10
        ret = 4 * (u_ * u_) * x / ((2 + x * x) ** (5.0 / 6)) / w / 2.0
    return ret

@numba.jit(nopython=True)
def davenport(v10, I10, alpha, f, z):
    """ <风荷载规范中若干问题的研究> 夏瑞光
    Davenport 1961, adopted in Chinese code
            S(n) = 4 * k * v10**2 * x**2 / n / (1 + x**2)**(4/3)
    where, k is the ground roughness coefficient, ;
           x = 1200 * n / v10;
           v10 is the mean wind speed at 10m height

    Args:
        v10 ([type]): [description]
        I10 ([type]): [description]
        alpha ([type]): [description]
        f ([type]): [description]
        z ([type]): [description]

    Returns:
        [type]: [description]
    """
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
def kaimal(v10, I10, alpha, f, z):
    """ <风荷载规范中若干问题的研究> 夏瑞光
    the expression:
            S(z, n) = 100 * x * sigma**2 / 3 / n / (1 + 50x)**(5/3)
    where, sigma is the turbulence intesisty, RETHINK: at 10m height or not?
           x = n * z / vz, vz is the mean wind speed at z height

    adopted in Europe code:
            S(z, n) = 6.8 * x * sigma**2 / n / (1 + 10.2x)**(5/3)
    where, x = n * Lu / vz, Lu = 300 * (z / 200) ** (0.67 + 0.05 * ln(z0)),
           z0 is the ground roughness length, taken from the follow table
           | terrain |  0  |  I  |  II  |  III  |  IV  |
           |  z0(m)  |0.003| 0.01| 0.05 |  0.3  |  1.0 |

    Args:
        v10 ([type]): [description]
        I10 ([type]): [description]
        alpha ([type]): [description]
        f ([type]): [description]
        z ([type]): [description]

    Returns:
        [type]: [description]
    """
    vz = v10 * (z / 10.0) ** alpha
    # x = f * z / vz
    Lu = 300 * (z / 200.0) ** (0.67 + 0.05 * np.log10(0.01))
    x = f * Lu / vz
    sigma = I10 # RETHINK: at 10m height or not?
    # k = 0.00129
    k = I10 * I10 / 6.0
    ret = 6.8 * x * k * (v10 ** 2) / f / ((1 + 10.2 * x) ** (5.0 / 3))
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
    vt = np.zeros((m, npts))
    for i in range(npts):
        for j in range(nfreq):
            for k in range(i+1):
                wml = j * dw + (1+k) * dw / npts
                tmp = 2.0 * np.sqrt(dw) * np.cos(wml * t + phi[k,j])
                vt[:,i] += np.abs(Hw[j,i,k]) * tmp
    return vt

def fft_synthesis(Hw, nfreq, m, dw, npts, phi, t):
    vt = np.zeros((m, npts))
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
    ax.plot(t, x, lw=1, c="black")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("wind speed (m/s)")
    ax.grid(True)
    plt.title("wind velocity of point " + str(pid))
    if path is None:
        plt.pause(5)
    else:
        figname = os.path.join(path, "wind_speed_of_point_" + str(pid) + ".svg")
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
    ax.loglog(fxx, pxx, c="Black", lw=0.8, label="simulated")
    ax.loglog(f, sf, c="red", lw=2, label="target")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("S(f) (m^2/s)")
    ax.legend()
    ax.grid(True)
    if path is None:
        plt.pause(5)
    else:
        figname = os.path.join(path, "spectrum_of_point_" + str(pid) + ".svg")
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
        self.num_time = waws.getint("number of segments of time")
        self.double_index = waws.getboolean("double indexing frequency (yes/no)")

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
        self.I10 = config.I10
        self.alpha = config.alpha
        self.spectrum = config.spectrum_type.lower()
        self.coherence = config.coh_type.lower()
        self.cx = config.cx
        self.cy = config.cy
        self.cz = config.cz

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

    def _spectrum(self):
        if self.points is None:
            raise UnboundLocalError("Does not exsit simulated points!")
        npts = len(self.points)
        self.Sw = np.zeros((len(self.wml), npts, npts))

        z = self.points[:,3]
        vz = self.v10 * (z / 10) ** self.alpha
        command = self.spectrum + "(self.v10, self.I10, self.alpha, self.wml, z[i])"
        for i in range(npts):
            self.Sw[:,i,i] = eval(command)

        if not self.double_index:
            self.target_Sw = self.Sw   # single indexing frequency
        else:
            self.target_Sw = self.Sw[:self.N,:,:]  # double indexing frequency

    def _coherence(self):
        npts = len(self.points)
        self.coh = np.zeros((len(self.wml), npts, npts))
        x = self.points[:,1]
        y = self.points[:,2]
        z = self.points[:,3]
        cx, cy, cz = self.cx, self.cy, self.cz
        z = self.points[:,3]
        vz = self.v10 * (z / 10) ** self.alpha

        # FIXME: change to circle frequency
        for i in range(npts):
            for j in range(npts):
                self.coh[:,i,j] = np.exp(-2.0 * self.wml / 2 / np.pi * np.sqrt(
                                  cx * cx * (x[i] - x[j]) ** 2 +
                                  cy * cy * (y[i] - y[j]) ** 2 +
                                  cz * cz * (z[i] - z[j]) ** 2) /
                                  (vz[i] + vz[j]))

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
        Hw = np.zeros_like(self.Sw)
        for i in range(len(self.wml)):
            Hw[i,:,:] = np.linalg.cholesky(self.Sw[i,:,:])

        self.Hw = np.zeros((self.N, npts, npts))
        if not self.double_index:
            self.Hw = Hw
        else:
            for i in range(npts):
                s, e = i * self.N, (i + 1) * self.N
                self.Hw[:,:,i] = Hw[s:e,:,i]

    def generate(self, method="fft"):
        """[summary]

        Args:
            method (str, optional): [description]. Defaults to "direct".
        """
        method = method.lower()
        if method not in ["fft", "deodatis"]:
            raise ValueError("unrecongnized method! fft or deodatis?")

        print("generate cross spectrum matrix...")
        self._spectrum()
        self._coherence()
        self._cross_spectrum_matrix()

        print("cholesky decompostion...")
        self._cholesky()
        npts = len(self.points)
        self.vt = np.zeros((self.M, npts))

        print("synthesis gust wind speed...")
        np.random.seed(0)
        phi = 2 * np.pi * np.random.rand(npts, self.N)
        dw = self.dw
        if method == "deodatis":
            self.vt = synthesis(self.Hw, self.N, self.M, dw, npts, phi, self.t)
        elif method == "fft":
            # TODO:
            self.vt = fft_synthesis(self.Hw, self.N, self.M, dw, npts, phi, self.t)
        else:
            raise ValueError("Unrecongnized method: " + method)
        print("finished!")
    
    def save(self):
        # create directory
        path = os.path.join(self.workdir, "results")
        if not os.path.exists(path):
            os.makedirs(path)

        # save wind speed
        fname = os.path.join(path, "wind_speed.csv")
        ans = np.hstack((self.t.reshape(-1,1), self.vt))
        print(self.points)
        head = ["t"] + [str(self.points[i,0]) for i in range(len(self.points))]
        np.savetxt(fname, ans, delimiter=",", header=",".join(head))

        # save traget spectrums
        fname = os.path.join(path, "target_spectrum.csv")
        npts = len(self.points)
        target = np.zeros((self.N, npts))
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

    def error(self):
        # create directory
        path = os.path.join(self.workdir, "results")
        if not os.path.exists(path):
            os.makedirs(path)

        # basic compare: spectrum compare
        for pid in self.target_PIDs:
            # check pid exists?
            ind = np.where(self.points[:,0]==pid)[0]
            if (ind.size == 0):
                raise Warning("Unvalid points ID: ", pid)
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
    gust1.generate(method="fft")
    start = time.time()
    # gust2.generate(method="deodatis")
    gust1.save()
    end = time.time()
    print("cost: ", end - start)
    gust1.error()

    # # compare fft and no-fft
    # fig, axs = plt.subplots(nrows=1, ncols=2)
    # axs = axs.flatten()
    # # time history
    # t1 = gust1.dt * np.arange(gust1.M)
    # # t2 = gust2.dt * np.arange(1, 1+gust2.M)
    # dt = gust1.dt
    # axs[0].plot(t1, gust1.vt[:,0], lw=1, c="black", label="fft")
    # axs[0].plot(t1, gust2.vt[:,0], lw=1, c="red", label="Deodatis", ls="dashed")
    # axs[0].grid(True)
    # axs[0].legend()

    # # spectrum
    # fs = 1 / dt
    # fxx1, pxx1 = signal.welch(gust1.vt[:,0], fs=fs, window="hann", 
    #                         nperseg=gust1.M, scaling="density")
    # fxx2, pxx2 = signal.welch(gust2.vt[:,0], fs=fs, window="hann",
    #                         nperseg=gust2.M, scaling="density")
    # # remove f > self.wup / 2 / pi
    # ind = np.where(fxx1 < gust1.wup / 2.0 / np.pi)
    # fxx1 = fxx1[ind]
    # pxx1 = pxx1[ind]
    # fxx2 = fxx2[ind]
    # pxx2 = pxx2[ind]
    # w = np.arange(gust1.dw, gust1.wup + gust1.dw, gust1.dw)

    # axs[1].loglog(fxx1, pxx1, lw=1, c="black", label="fft")
    # axs[1].loglog(fxx2, pxx2, lw=1, c="red", label="Deodatis", ls="dashed")
    # axs[1].loglog(w/2/np.pi, gust1.target[:,0,0]*2*np.pi, c="green", lw=1)
    # axs[1].grid(True)
    # axs[1].legend()

    # plt.show()
    # plt.close(fig)
