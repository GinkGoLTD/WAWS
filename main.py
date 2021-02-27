import os
import numpy as np
import scipy
import waws


if __name__ == "__main__":
    config = waws.ConfigData("config.ini")

    # npts = 5
    # points = np.zeros((npts,4))
    # for i in range(1,npts+1):
    #     points[i-1, 0] = i
    #     points[i-1, 3] = i * 10

    # gust = GustWindField(config, points)
    gust = waws.GustWindField(config)
    gust.generate(mean=True, method="fft")
    gust.save()
    gust.error()