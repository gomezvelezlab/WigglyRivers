# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 2022-03-24
# _____________________________________________________________________________
# _____________________________________________________________________________
"""

The functions given on this package allow the user to manipulate and create
functions from the computer.


"""
# ------------------------
# Importing Modules
# ------------------------
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import signal


# ------------------------
# Functions
# ------------------------

def savgol_filter(x, ds, order, savgol_window, kernel):
    """

    This function performs the savgol filter with a Gaussian filter.
    It is based in the functions presented in the pynumdiff package, created
    by Van Breugel et al. (2022).

    Van Breugel, F. V., Liu, Y., Brunton, B. W., & Kutz, J. N. (2022).
    PyNumDiff: A Python package for numerical differentiation of noisy
    time-series data. Journal of Open Source Software, 7(71), 4078.
    https://doi.org/10.21105/joss.04078


    :param x: np.array,
        Coordinates to filter
    :type x: np.ndarray
    :param ds: float,
        Difference in distance.
    :type ds: float
    :param order: int,
        Polynomial order in the savgol filter
    :type order: int
    :param savgol_window: int,
        Savgol filter window. It has to be an odd number if an even number
        is given the function will sum one to the value.
    :type savgol_window: int
    :param kernel: np.array,
        Gaussian kernel to apply additional smoothing to the function.
    :type kernel: np.ndarray
    :return: x_smooth: smoothed function
    :rtype:
    """
    # -----------------
    # Apply Filter
    # -----------------
    if savgol_window < order:
        raise ValueError(f"savgol_window must be larger than poly_order")
    dxds = signal.savgol_filter(x, savgol_window, order, deriv=1) / ds

    # ------------------------
    # Apply Gaussian Smoother
    # ------------------------
    dxds_smooth = convolution_smoother(dxds, kernel, 1)

    # ------------------------
    # Integrate Solution
    # ------------------------
    x_smooth = integrate.cumtrapz(dxds_smooth)
    first_value = x_smooth[0] - np.mean(dxds_smooth[0:1])
    x_smooth = np.hstack((first_value, x_smooth)) * ds

    # Find the integration constant that best fits the original coordinates
    def f(x0, *args):
        x, x_smooth = args[0]
        error = np.linalg.norm(x - (x_smooth + x0))
        return error

    result = optimize.minimize(f, [0], args=[x, x_smooth], method='SLSQP')
    x_0 = result.x
    x_smooth = x_smooth + x_0

    return x_smooth


def gaussian_function(t, sigma):
    result = 1 / np.sqrt(
        2 * np.pi * sigma ** 2) * np.exp(-(t ** 2) / (2 * sigma ** 2))
    return result


def convolution_smoother(x, kernel, iter):
    """
    Calculates a mean smoothing by convolution, This function is based
    on the __convolution_smoother__ of pynumdiff, created by 
    Van Breugel et al. (2022).

    Van Breugel, F. V., Liu, Y., Brunton, B. W., & Kutz, J. N. (2022).
    PyNumDiff: A Python package for numerical differentiation of noisy
    time-series data. Journal of Open Source Software, 7(71), 4078.
    https://doi.org/10.21105/joss.04078


    :param x: np.array 1xN,
        Coordinates to differentiate
    :type x: np.ndarray
    :param kernel: np.array (1xwindow_size),
        Kernel used in the convolution
    :type kernel: np.ndarray
    :param iter: int,
        Number of iterations >= 1
    :type iter: int
    :return: x_smooth:
            Smoothed x
    :rtype: ndarray
    """
    x_smooth = np.hstack((x[::-1], x, x[::-1]))
    for _ in range(iter):
        x_smooth_f = np.convolve(x_smooth, kernel, 'same')
        x_smooth_b = np.convolve(x_smooth[::-1], kernel, 'same')[::-1]

        w = np.arange(0, len(x_smooth_f), 1)
        w = w / np.max(w)
        x_smooth = x_smooth_f * w + x_smooth_b * (1 - w)

    return x_smooth[len(x): len(x) * 2]

