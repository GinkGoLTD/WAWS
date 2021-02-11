import os
from configparser import ConfigParser



class ConfigData(object):
    def __init__(self, fname):
        if not os.path.exists(fname):
            raise ValueError(fname + " does not exist!")
        self.config = ConfigParser()
        self.config.read(fname, encoding="utf-8")
        self.parser()

    def _paser_wind(self):
        wind = self.config["wind"]
        ref_wind_speed = wind.getfloat("reference wind speed (m/s)")
        ref_turbulence_intensity = wind.getfloat("reference turbulence intensity")
        self.v10 = ref_wind_speed
        self.I10 = ref_turbulence_intensity

    def parser(self):
        self._paser_wind()


if __name__ == "__main__":
    fname = "config.ini"
    # config = ConfigParser()
    print(os.getcwd())
    config = ConfigData(fname)
    print(config.I10, config.v10)
    # # config.read(fname, encoding="utf-8")
    # sections = config.sections()
    # print(config.sections())
    # print(config.items(sections[-1]))
    # points = config["points"]
    # print(points)
    # num_points = int(points.get("number of points"))
    # print(num_points)
    # print(type(num_points))
