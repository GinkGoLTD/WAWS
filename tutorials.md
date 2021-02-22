---
typora-copy-images-to: documents/figures
---



# 谐波合成法(WAWS)程序使用手册

以功率谱作为权系数，与一系列带随机相位的三角级数的加权和来逐渐逼近随机过程，适用于指定谱特征的平稳高斯随机过程。该方法有恒幅谐波叠加法和加权振幅谐波叠加法(尚未阅读相关文献). 


## 1. 谐波合成法基本步骤

**Step 1: ** 选取目标风速谱$S(w)$和相干函数$coh(x_i, x_j, y_i, y_j, z_i, z_j, f)$,确定模拟点坐标

**Step 2: **生成互谱密度矩阵
$$
S(w) = \begin{bmatrix} S_{11}(\omega) &  S_{12}(\omega)  ...  & S_{1n}(\omega) \\
				   S_{21}(\omega) &  S_{22}(\omega)  ...  & S_{2n}(\omega) \\
				   . & ........ & . \\
				   . & ........ & . \\
				   . & ........ & . \\
				   S_{n1}(\omega) &  S_{n2}(\omega)  ...  & S_{nn}(\omega) \\\end{bmatrix}\\
				   
S_{ij}(\omega)=\sqrt{ S_{12}(\omega) S_{12}(\omega)}coh(x_i,x_j,y_i,y_j,z_i,z_j,\omega)
$$

**Step 3: **对互谱密度矩阵进行cholesky分解

$$
S(w) =H(w) H^T(w)\\
H(w) = \begin{bmatrix} H_{11}(\omega) &  0 ...  & 0 \\
				   H_{21}(\omega) &  H_{22}(\omega)  ...  & 0 \\
				   . & ........ & . \\
				   . & ........ & . \\
				   H_{n1}(\omega) & H_{n2}(\omega)  ...  & H_{nn}(\omega) \\\end{bmatrix}
$$

**Step 4: (option 1):  ** M. Shinuzuka提出的单索引频率法

这种算法最简单, 只需互谱密度矩阵进行N次cholesky分解(nxn阶矩阵). 但是由于频率均匀分布可能导致模拟曲线出现周期性[^1]. (*具体情况尚未进行过详细地分析, 有机会在做吧, 暂且相信文献的结论*)
$$
V_j(t) = 2\sqrt(\Delta \omega) \sum_{m=1}^{j} \sum_{l=1}^{N}|H_{jm}(\omega_{l})cos(\omega_l t + \theta_{jm}(\omega_l) + \phi_{ml})\\
\omega_l = l\Delta\omega, l = 1,2,3,...,N
$$

**Step 4 (option 2): ** Deodatis双索引频率法计算模拟点风速时程

公式来源于<基于实测风特性的台风模拟>(杨素珍)[^2]
由于option 1可能存在模拟风速时程出现周期性的问题, 引入了双索引频率, 将频率微小扰量均匀分布在频率增量内, 这样既可以采用FFT算法, 又能保证模拟曲线的各态历经性. 但互谱密度矩阵的分解数量将明显提升, 达到$N \times n$次($N$为频率等分数, $n$为模拟点数), 这样运算量将会增加; (但似乎采用`numpy.linalg.cholesky()`函数计算效率并不低)
$$
V_j(t) = 2\sqrt{\Delta\omega}\sum_{m=1}^{j}\sum_{l=1}^{N}|\H_{jm}(\omega_{ml})| cos(\omega_{ml}t - \theta(\omega_{ml}) + \phi_{ml})\\
\phi_{ml}=rand[0,2\pi]\\
\theta_{jm}=arctan[\frac{Im[H_{jm}(\omega_{ml})]}{Re[H_{jm}(\omega_{ml})]}]\\
\omega_{ml} = (l-1)\Delta\omega + m/n\Delta\omega, l=1,2,3,...,N
$$
**注意:** 有的文献中将$2\sqrt{\Delta\omega}​$写成$\sqrt{2\Delta\omega}​$这是错的
**注意:** 如果$S(w)​$为实正定矩阵,则$H(w)​$也为实矩阵,因此$\theta_{jm}=0$

实际上, 在采用双索引算法时,并非$H(\omega_{ml})$矩阵中的所有元素都将用于最终的谐波叠加. 保留$H(\omega_{ml}$矩阵中用到的元素, 组成如下有效元素矩阵

$$
H^{eff}(w) = \begin{bmatrix} H_{11}(\omega_{1l}) &  0 ...  & 0 \\
H_{21}(\omega_{1l}) &  H_{22}(\omega_{2l})  ...  & 0 \\
. & ........ & . \\
. & ........ & . \\
H_{n1}(\omega_{1l}) & H_{n2}(\omega_{2l})  ...  & H_{nn}(\omega_{nl}) \\\end{bmatrix}
$$


**采用这种方式合成脉动风速需要多次累加计算, 模拟非常耗时, 尤其是当模拟点数较多时**

**Step 4 (option 3): ** 采用FFT技术加速风速时程的合成

公式来源于<基于实测风特性的台风模拟>(杨素珍)[^2]
实际计算过程中发现整个模拟过程中合成风速时程是比较耗时的, 而采用FFT技术可以显著提升模拟速度
$$
V_j(t) = 2\sqrt{\Delta\omega}\sum_{m=1}^{j}\sum_{l=1}^{N}|\H_{jm}(\omega)| cos(\omega_{ml}t - \theta(\omega_{ml}) + \phi_{ml})\\
$$

引入欧拉公式$e^{ix} = cosx + isinx$,且$\theta_{jm}(\omega_{wl}=0$, 上式可以改写为
$$
V_j(t) = 2\sqrt{\Delta\omega}\sum_{m=1}^{j}\sum_{l=1}^{N}|\H_{jm}(\omega_{ml})| Re[e^{i(\omega_{ml}t + \phi_{ml})}]\\
= Re[2\sqrt{\Delta\omega}\sum_{m=1}^{j}\sum_{l=1}^{N}|\H_{jm}(\omega_{ml})| e^{i\omega_{ml}t} e^{i\phi_{ml}}]\\
=Re[\sum_{m=1}^{j}(\sum_{l=1}^{N} 2 \sqrt{\Delta\omega} |H_{jm}(\omega_{ml})| e^{i\omega_{ml}t} e^{i\phi_{ml} })]
$$

代入$\omega_{ml} = (l-1)\Delta\omega + \frac{m}{n}\Delta\omega$, 上式可以改写为
$$
V_j(t) = Re[\sum_{m=1}^{j}(\sum_{l=1}^{N} 2 \sqrt{\Delta\omega} |H_{jm}(\omega_{ml})| e^{i((l-1)\Delta\omega + \frac{m}{n}\Delta\omega)t} e^{i\phi_{ml} })]\\
=Re[\sum_{m=1}^{j}(\sum_{l=1}^{N} 2 \sqrt{\Delta\omega} e^{i\phi_{ml}} |H_{jm}(\omega_{ml})| e^{i((l-1)\Delta\omega t} )e^{i\frac{m}{n}\Delta\omega t} ]\\
= Re[\sum_{m=1}^{j}G_{jm}(p\Delta t) e^{i\frac{m}{n} \Delta \omega t}]
$$

注意, 上式中括号内部分刚好是离散傅里叶逆变换的表达式
$$
G_{jm} = (\sum_{l=1}^{N} 2 \sqrt{\Delta\omega} e^{i\phi_{ml}} |H_{jm}(\omega_{ml})| e^{i(l-1)\Delta\omega t} )\\
$$

令$k = l - 1, (k=0,1,2,...,N-1)$ 且 , $B_{jm} = 2 \sqrt{\Delta\omega} e^{i\phi_{ml}} |H_{jm}(\omega_{ml})|, \Delta\omega=\frac{\omega}{N}=\frac{2\pi f}{N}$,上式可以改写为
$$
G_{jm}(p\Delta t) = (\sum_{l=1}^{N}B_{jm} e^{ik\Delta\omega t} ) = (\sum_{k=0}^{N-1}B_{jm} e^{i k \Delta \omega p\Delta t} )
$$

信号的最长周期$T_0 = \frac{2\pi}{\Delta \omega}$, 若采样时间间隔为$\Delta t$, 一个周期内的时间点数为$M = \frac{T_0}{\Delta t}$, 则
$$
T_0 = M \Delta t = \frac{2 \pi}{\Delta \omega}\\
\Delta t \Delta \omega = \frac{2 \pi}{M}
$$
带入上上式中
$$
G_{jm}(p\Delta t) = (\sum_{l=1}^{N}B_{jm} e^{ik\Delta\omega t} ) = (\sum_{k=0}^{M-1}B_{jm} e^{i kp \frac{2\pi}{M}})\\
B_{jm} = \left \{ 
\begin{aligned}
2 \sqrt{\Delta\omega} e^{i\phi_{ml}} |H_{jm}(\omega_{ml})|, 0 \leq l < N \\
0,  N \leq < l <M
\end{aligned}
\right.
$$
而离散傅里叶逆变换的表达式为$x(t) = \frac{1}{N} \sum_{n=0}^{N-1} X(n) e^{i\frac{2 \pi}{N} t n}$ 因此, 可以对$B_{jm}$进行离散傅里叶逆变换从而提高运算效率, (FFT的时间复杂度为$nlog(n)$). 



**说明:** 在很多文献中对这部分都描述为使用FFT技术加速计算, 然而并未提及到底是使用`fft`还是`ifft`; 另一方面在网络上看到的很多代码中, 都是直接使用`fft`进行计算的. 然而, 依据上述推导过程及表达式, 显然应该使用`ifft`才比较合理; 另一方面, 从物理意义上看, `ifft`是将信号从频域转换到时域, 和$B_{jm}转换到G_{jm}$这个过程也是吻合的. 此外, 为了验证采用`ifft`的正确性, 对比了直接采用Deodatis双索引公式合成的脉动风和采用`ifft`合成的脉动风速的时程和频谱, 结果表明二者完全一致, 这足以说明采用`ifft`的正确性.

![Fig2](documents/figures/Fig2.png)

## 2. 关于一些参数的选择

### 2.1 关于模拟时间间隔$\Delta t$的选取

主要参考文献: <谐波合成法脉动风模拟时间步长的取值>(张军锋)[^3]

#### 2.1.1 频率点数$N$的选取原则

若模拟时将频率划分为$N$等份,则合成脉动风场时的频率分辨率为$\Delta\omega=\omega_u/N$.  显然,$N$越大, $\Delta\omega$越小, 模拟结果越精确, 根据Shinozuka[^4]的研究当$N>1000$ 之后模拟结果已经足够精确. (需要说明的是, 这个值显然需要考虑$\omega_u$的大小, $N$的选取原则还是以$\Delta\omega$足够小来控制). $N$越大意味着频率分辨率越高, 结果越能反应目标谱的频率分布特征, 然而$N$越大, 需要进行cholesky分解的次数就越多, 合成风速时程的运算量也会相应增加, 模拟越耗时.

#### 2.1.2 时间点数$M$的选取原则

**采用WAWS生成的脉动风场可以理解为一个模拟信号$V_j(t)$(该信号在时间上连续的), 这个信号的频率上限为$\omega_{u}$, 包含$n\omega_{u}/N, n=0,1,2,...,N$个频率成分. 而最终使用的是从$V_j(t)$采样得到的离散的数字信号,所以应该遵从采样定理.**

根据合成公式, 生成的信号在每个频率点的周期为$T_l = 2 \pi / \omega_l = 2 \pi N / l \omega_u$, 叠加后的脉动风场也是周期函数, 其中的最长周期为
$$
T_0 = \frac{2 \pi}{\omega_1} = \frac{2 \pi}{\Delta \omega} = \frac{2 \pi N}{\omega_u}
$$
定义$M​$为最长周期内的时间点数量, 即
$$
T_0 = M \Delta t = \frac{2 \pi} {\Delta \omega} = \frac{2 \pi N}{\omega_u}\\
\Delta t = \frac{2 \pi} {\Delta \omega M} = \frac{2 \pi N}{\omega_u M} = \frac{1}{a} \frac{2 \pi}{\omega_u}, a =\frac{M}{N}
$$
 

根据采样定理, 为了避免频率混叠, 从$V_{j}(t)$采样的采样频率必须满足如下关系
$$
f_{s} \geq 2 f_{u} = 2 \frac{\omega_{u}}{2\pi}=\frac{\omega_{u}}{\pi}
$$
因此
$$
\Delta t = \frac{1}{f_s} \leq \frac{\pi}{\omega_u}
$$
**注意:**WAWS生成的模拟信号所包含的最高频率成分也仅为$\omega_u$, 采样得到的数字信号中的频率成分为$[0, \frac{f_s}{2}]$, 而其中高于$\omega_u=2\pi f_u$的部分都是虚假的(无效的)



模拟信号总持时:
$$
T_0 = M \Delta t
$$
根据上述公式, $\Delta t​$的选取只要满足采样定理就一定能保证生成的脉动风的功率谱密度与目标谱一致. 然而, 考虑到模拟的脉动风速时程主要用于结构动力时程分析, 因此$\Delta t​$ 的选择还应该考虑是否会影响结构计算的精度, 张军锋[^2]针对这个问题进行了研究, 在此只给出他的研究结论:

1. $M/N \geq 2$后, $M/N$比值并不影响生成的脉动风功率谱密度与目标谱的一致性;但对应的$\Delta t$可能在动力时程计算中无法准确计入结构的共振响应,尤其当需要计及的结构频率在$\omega_u$附近时. 且如果$\Delta t$过大, 即使在动态时程分析时增加荷载子步(阶跃荷载或线性插值,原文采用ANSYS进行分析)也无法准确计入共振效应,必须减小模拟脉动风的采样间隔$\Delta t$
2. 对单自由度系统而言, 当$M/N \geq 8$之后, 通过模拟风速时程计算的系统响应与理论解非常接近, 可以作为$M/N$取值的参考; 单自由度弹簧振子模型对$\Delta t$的要求最为苛刻, 可以作为普通结构的上限

#### 2.1.3 模拟中时频参数选取建议和实施步骤

根据上述分析, 关于$M, N$的选取可以按照如下步骤执行

- **Step 1: 合理确定周期$T_0$(模拟风速总时长$T$), 一般有$T_0 \geq T = 600 s$**
  - 通过WAWS生成的脉动风荷载主要用于结构的斗阵分析, 因而10min以上的平均才具有统计意义
  - *对于非平稳风场,还需了解更多文献*

- **Step 2: 确定截止频率上限$\omega_u = 2 \pi f_u$, $f_u \geq f_{struct}$**
    - $f_{struct}$为所关心的结构频率. 如对输电塔而言, $f_{struct}$可以取1阶扭转频率, 但如果需要计入高阶模态的影响,则应该考虑更高阶模态频率
- **Step 3: 确定频率点$N​$, 通常可以取 $N = 1024​$ 或$N=2048​$ (确保$N >1000​$)**
  - 注意$N$越大模拟越精确,但是也更耗时
  - 频率点数$N$直接影响生成的脉动风信号的最长周期$T_0=\frac{2 \pi N}{\omega_u}$ , 因此选择$N$需要考虑到$T_0 > T$
- **Step 4: 确定模拟时间点, 可以取$M = 8N$(确保$M\geq2N$)**

按照上述步骤确定相关参数取值后:
$$
\Delta \omega = \frac{\omega_u}{N} \\
\Delta t = \frac{T_0}{M}
$$
### 2.2 关于角频率$\omega $和频率$f$, 频谱, 功率谱等

角频率或称圆频率$\omega (rad/s)$与频率$f (Hz)$之间存在如下关系:
$$
\omega = 2 \pi f
$$
如果输入的风谱和相干函数为频率的函数,即$S(f), coh(f)$, 那么经过cholesky分解后得到的下三角矩阵也应该为频率的函数$H(f)$, 因而在合成脉动风速时, 应该按照频率进行叠加
$$
V_j(t) = 2\sqrt{\Delta f}\sum_{m=1}^{j}\sum_{l=1}^{N}\lvert\H_{jm}(f) \rvert cos(\omega_{ml}t - \theta(\omega_{ml}) + \phi_{ml})
$$
而$\Delta f = \Delta \omega / (2\pi)$



另一方面, 如果输入的风谱和相干函数为角频率的函数,即$S(w), coh(w)$, 那么采用则完全按照角频率的公式进行计算.  然而需要注意的是, 对生成的脉动风信号进行频谱分析时(采用scipy.signal.welch), 得到的是S(f); 而目标谱则为$S(w)$, 进行比较时需将二者统一.

#### 2.2.1 两种频率表示的频谱$F(\omega)$和$F(f)$

若存在随机信号$f(t), t\in (-\infin, \infin)$ 满足狄氏条件(信号存在傅里叶变换的**充分不必要**条件), 且绝对可积
$$
\int_{-\infin}^{\infin}|f(t)|dt < \infin
$$
$f(t)$的傅里叶逆变换为
$$
f(t) = \frac{1}{2\pi} \int_{-\infin}^{\infin}F(\omega)e^{j\omega t}d\omega = \int_{-\infin}^{\infin}F(f) e^{j2\pi ft}df \\

傅里叶变换:\\
F(\omega) = \int_{-\infin}^{\infin}f(t)e^{-jwt}dt \\
F(f) = \int_{-\infin}^{\infin}f(t) e^{-j2\pi ft}dt
$$
可以发现, 信号$f(t)$的频谱用$\omega$和$f$表示时存在一个压缩关系, 相差$2\pi$倍, 即$F(\omega)$和$F(f)$并不相等

#### 2.2.2 两种频率表示的功率谱密度$P(\omega)$和$P(f)$

从能量角度出发(信号满足Parseval定理), 信号在一个周期内的总能量如下
$$
E_T = \int_{-\infin}^{\infin}f^2(t)dt = \frac{1}{2\pi} \int_{-\infin}^{\infin}|F(\omega)|^2d\omega = \int_{-\infin}^{\infin}|F(f)|^2df
$$
上式的含义是, 信号在时域内的能量与频域内的能量相等

那么一个周期内的平均功率如下
$$
P = \lim\limits_{T\to\infin} \int_{-T/2}^{T/2} \frac{1}{T} f^2(t)dt = \frac{1}{2\pi} \int_{-\infin}^{\infin} \lim\limits_{T\to\infin} \frac{1}{T}|F(\omega)|^2d\omega = \int_{-\infin}^{\infin} \lim\limits_{T\to\infin} \frac{1}{T} |F(f)|^2df
$$
$f(t)$的功率谱
$$
P(\omega) = \lim\limits_{T\to\infin} \frac{|F(\omega)|^2}{2\pi T} \\
P(f) = \lim\limits_{T\to\infin} \frac{|F(f)|^2}{T}
$$
注意: $P(\omega)$与$P(f)$并不相等
$$
P = \int_{-\infin}^{\infin}P(\omega)d\omega = \int_{-\infin}^{\infin} P(f)df\\
P(f) = 2\pi P(\omega)
$$
![Fig1](documents/figures/Fig1.jpg)

$$PSD = \lim_{T\to\infty}\frac{E[|\hat{f_{T}}(\omega)|^2]}{2T}​$$

正如示意图所示, 同一信号的功率谱密度$P(f)$和$P(\omega)$所蕴含的总能量应该是相等的

另一方面从相关关系的角度也能得出相同的关系
$$
P(\omega) = \frac{1}{2\pi} \int Rxx(\tau) e^{-j\omega \tau} d\tau \\
P(f) =\int Rxx(\tau) e^{-j2\pi f\tau} d\tau \\
P(f) = 2\pi P(\omega)
$$
#### 2.2.3 单边谱与双边谱
现实世界不存在负频率
$$
P_{oneside}(\omega) = 2P_{twoside}(\omega)
$$

#### 2.2.4 程序中关于风谱的使用说明
1. 考虑到上述谐波合成法理论中, 风谱均表示为角频率$\omega$的函数, 因此输入的目标功率谱也均采用$S_v(\omega)$, 在使用自编的风谱函数时,请务必注意
2. 为了减少计算量, 按照风谱函数计算的目标将存储在`target`属性中, 对比生成信号功率谱和目标功率谱一致性时, 将直接调用这个数据
3. 生成信号的功率谱采用welch方法计算(`scipy.singal.welch`), 得到的结果是$S_v(f)$; 因此将`target`中存储的数据转换后进行对比, 具体方法为: 横坐标$f=\omega/(2\pi)$, 纵坐标: $S_v(f) = target * 2  \pi$; (具体原理见2.2节)


### 2.3 Deodatis双索引和基于FFT的风速时程合成

根据文献[^1]Deodatis引入双索引频率的目的在于减小模拟样本的周期性, 同时可以实现生成的脉动风场的各态历经性.  从下面的表达式可以看出, 如果
$$
V_j(t) = 2\sqrt(\Delta \omega) \sum_{m=1}^{j} \sum_{l=1}^{N}|H_{jm}(\omega_{l})cos(\omega_l t + \theta_{jm}(\omega_l) + \phi_{ml})\\
\omega_l = l\Delta\omega, l = 1,2,3,...,N
$$
然而引入双索引频率$\omega_{ml}, m=1, 2, ..., npts; l = 1, 2, 3, ..., N$后, `cholesky`分解的运算量将从$N$次陡增到$npts \times N$次. 


## References

[^1]: <大跨度桥梁脉动风场模拟的插值算法>(祝志文)
[^2]: <基于实测风特性的台风模拟>(杨素珍)
[^3]: <谐波合成法脉动风模拟时间步长的取值>(张军锋)
[^4]: Shinozuka M, Deodatis G. Simulation of stochastic processes by spectral representation[J]. Applied mechanics reviews. 1991, 44(4): 191-204

