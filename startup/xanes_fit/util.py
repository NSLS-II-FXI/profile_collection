from numpy.polynomial.polynomial import polyfit, polyval
import numpy as np
from scipy.interpolate import interp1d

def find_nearest(data, value):
    return np.abs(data - value).argmin()


def fit_curve(x_raw, y_raw, x_fit, deg=1):
    coef1 = polyfit(x_raw, y_raw, deg)
    y_fit = polyval(x_fit, coef1)
    return y_fit