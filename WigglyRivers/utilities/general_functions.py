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
from typing import Union
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import signal
from scipy.spatial import Delaunay


# ------------------------
# Functions
# ------------------------


def savgol_filter(
    x: np.ndarray, ds: float, order: int, savgol_window: int, kernel: np.ndarray
) -> np.ndarray:
    """This function performs the savitzky golay filter with a Gaussian filter.

    It is based in the functions presented in the pynumdiff package, created
    by Van Breugel et al. (2022).

    Van Breugel, F. V., Liu, Y., Brunton, B. W., & Kutz, J. N. (2022).
    PyNumDiff: A Python package for numerical differentiation of noisy
    time-series data. Journal of Open Source Software, 7(71), 4078.
    https://doi.org/10.21105/joss.04078

    Args:
        x (np.ndarray): coordinates to filter.
        ds (float): delta in distance.
        order (int): order of the polynomial in the savgol filter.
        savgol_window (int): savgol filter window. It has to be an odd number
            if an even number is given the function will sum one to the value.
        kernel (np.ndarray): Gaussian kernel to apply additional smoothing to
            the function.

    Raises:
        ValueError: if savgol_window is smaller than order.

    Returns:
        np.ndarray: smoothed values of the coordinates.
    """
    # -----------------
    # Apply Filter
    # -----------------
    if savgol_window < order:
        raise ValueError("savgol_window must be larger than poly_order")
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

    result = optimize.minimize(f, [0], args=[x, x_smooth], method="SLSQP")
    x_0 = result.x
    x_smooth = x_smooth + x_0

    return x_smooth


def gaussian_function(
    t: Union[float, np.ndarray], sigma: Union[np.ndarray, float]
) -> Union[float, np.ndarray]:
    """Gaussian function

    Args:
        t (Union[float, np.ndarray]): array of values to evaluate the function.
        sigma (Union[np.ndarray, float]): array of values of the standard
            deviation.

    Returns:
        Union[float, np.ndarray]: array of values of the gaussian function.
    """

    result = (
        1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(t**2) / (2 * sigma**2))
    )
    return result


def convolution_smoother(
    x: np.ndarray, kernel: np.ndarray, iter: int
) -> np.ndarray:
    """Calculates a mean smoothing by convolution.

    This function is based
    on the __convolution_smoother__ of pynumdiff, created by
    Van Breugel et al. (2022).

    Van Breugel, F. V., Liu, Y., Brunton, B. W., & Kutz, J. N. (2022).
    PyNumDiff: A Python package for numerical differentiation of noisy
    time-series data. Journal of Open Source Software, 7(71), 4078.
    https://doi.org/10.21105/joss.04078

    Args:
        x (np.ndarray): 1xN, Coordinates to differentiate
        kernel (np.ndarray): 1xwindow_size, Kernel used in the convolution.
        iter (int): Number of iterations >= 1.

    Returns:
        np.ndarray: coordinates smoothed.
    """
    x_smooth = np.hstack((x[::-1], x, x[::-1]))
    for _ in range(iter):
        x_smooth_f = np.convolve(x_smooth, kernel, "same")
        x_smooth_b = np.convolve(x_smooth[::-1], kernel, "same")[::-1]

        w = np.arange(0, len(x_smooth_f), 1)
        w = w / np.max(w)
        x_smooth = x_smooth_f * w + x_smooth_b * (1 - w)

    return x_smooth[len(x) : len(x) * 2]


def circumcenter(tri: Delaunay) -> np.ndarray:
    """Compute the circumcenter of a triangle. The point where the
    perpendicular bisectors of the sides of the triangle intersect.

    Args:
        tri (Delaunay): Delaunay triangulation.

    Returns:
        np.ndarray: circumcenter of the triangle.
    """
    # Get the indices of the vertices that form the triangle
    tri_indices = tri.simplices[0]

    # Get the coordinates of the vertices that form the triangle
    tri_coords = tri.points[tri_indices]

    # Calculate the circumcenter of the triangle
    a = tri_coords[0]
    b = tri_coords[1]
    c = tri_coords[2]
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    x = (
        (a[0] ** 2 + a[1] ** 2) * (b[1] - c[1])
        + (b[0] ** 2 + b[1] ** 2) * (c[1] - a[1])
        + (c[0] ** 2 + c[1] ** 2) * (a[1] - b[1])
    ) / d
    y = (
        (a[0] ** 2 + a[1] ** 2) * (c[0] - b[0])
        + (b[0] ** 2 + b[1] ** 2) * (a[0] - c[0])
        + (c[0] ** 2 + c[1] ** 2) * (b[0] - a[0])
    ) / d
    cc = np.array([x, y])
    return cc
