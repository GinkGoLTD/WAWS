# WAWS

#### 介绍

WAWS is a generalized solver for wind simulation by utilizing **weighted  Amplitude Wave Superpostion method**. Although it is a quite mature and wide-used method, there is no avaible python module for it.  



WAWS is developed as an open-source code with the following objectives:

- an open, well-documented implementation of the WAWS models for modeling stationary gust wind field;
- be capable of simulated several wind spectrum, such as Davenport, Harris and Simiu etc.;

I hope that this program will be used by research laboratories, academia, and industry to simulated gust wind fields.

 The program is still under development. New featrues will be added:

- [ ] The interpolation algorithm will be introduced to  acclerate the decompositon of cross-spectrum matrix
- [ ] Non-stationay gust wind simulation
- [ ] Other wind simulation method



I am still struggling with my Doctoral thesis, the new featrue will developed in futrue.



#### 软件架构
The program is consisted of several spectrum functions, helper functions and two classes (`ConfigData, GustWindField`).  All the simulating parameters are inputed in "config.ini" file, and the `ConfigData` class is implemented to parse the the "config.ini" file.  And the `GustWindField` class is in charge of the whole process of the WAWS. 

  

#### 安装教程

1. make sure you have installed the following modules:

   - numpy
   - scipy
   - matplotlib
   - numba

2. don not need to install, just download and copy to your project directory, and

3. `import waws`

   

#### 使用说明

it is easy to use. 

``` python
import waws


if __name__ == "__main__":
    config = waws.ConfigData("config.ini")
    gust = waws.GustWindField(config)
    gust.generate(mean=True, method="fft")
    gust.save()
    gust.error()
```



#### 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request


#### 
