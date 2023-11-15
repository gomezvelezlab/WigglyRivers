# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2021-04-18
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
______________________________________________________________________________

 DESCRIPTION:
   Scripts related to meander creation and fitting.
______________________________________________________________________________
"""
# -----------
# Libraries
# -----------
from typing import Union, List, Tuple, Dict, Any, Optional
import time
import copy
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy.signal import find_peaks
from collections import Counter
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean

# Package packages
from ..utilities import general_functions as GF
from ..utilities.classExceptions import *
from ..utilities import filesManagement as FM
from ..wavelet_tree import WaveletTreeFunctions as WTFunc


# -----------
# Functions
# -----------
def convert_str_float_list_vector(x_val):
    """
    Description:
    ------------
        Convert a string with a list of values to a list of floats.
    ____________________________________________________________________________

    Args:
    ------------
    :param x_val: str,
        String with the list of values.
    :type x_val: str
    :return:
        x_val: np.ndarray, List of floats.
    """
    x_val = x_val.replace('[', '').replace(']', '').replace('\n', ',').replace(
        ' ', ',').split(',')
    x_val = np.array([float(x) for x in x_val if x != ''])
    return  x_val


def line_intersection(line1, line2):
    """
    Description:
    ------------
        Finds line intersection.
    ____________________________________________________________________________

    Args:
    ------------
    :param line1: np.ndarray,
        Line 1.
    :param line2: np.ndarray,
        Line 2.
    :return:
        x: float, Location where it intersects in x.
        y: float, Location where it intersects in y.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def kinoshita_curve(theta_0: float, lambda_value: float, j_s: float,
                    j_f: float, n: int, m_points: int=1000,
                    ds: Union[None, float]=None):
    """
    Description:
    ------------

        Generate a Kinoshita Curve with the information related
        to the reach generated.
    ________________________________________________________________

    Args:
    ------------
    :param theta_0: float,
        Maximum angular amplitude in radians.
    :type theta_0: float
    :param lambda_value: float,
        Arc wavelength.
    :type lambda_value: float
    :param j_s: float,
        Skewness.
    :type j_s: float
    :param j_f: float,
        Flatness.
    :type j_f: float
    :param n: int,
        Number of loops.
    :type n: int
    :param m_points: int (default 1000),
        Number of points that describe the meander.
    :type m_points: int
    :param ds: float or None (default None),
        Delta of streamwise coordinate (s).
    :type ds: float or None
    :return:
        - x (numpy.ndarray) - X coordinates.
        - y (numpy.ndarray) - Y coordinates.
        - data (dict) - dict with 'curve': curvature, 'theta':
        values of theta in each iteration, 's': streamwise coordinates,
        'lmax': maximum length, 'ymax': maximum y extent,
        'sinuosity': sinuosity (sigma = smax/lmax).
    :rtype: (numpy.ndarray, numpy.ndarray, py:class:dict)
    """
    # Direction
    smax = n * lambda_value
    if ds is None:
        s = np.linspace(0, smax, m_points)
    else:
        s = np.arange(0, smax + ds, ds)
    deltas = s[1]

    k = 2 * np.pi / lambda_value

    theta_rad = (
        theta_0*np.sin(k*s) +
        theta_0**3*(j_s * np.cos(3 * k * s) - j_f * np.sin(3 * k * s)))
    theta_rad = theta_rad[:-1]
    theta = theta_rad*180/np.pi

    curve = (k * theta_0 * np.cos(k*s) - 3 * k * theta_0 ** 3
             * (j_s * np.sin(3 * k * s)) +
             j_f * np.cos(3 * k * s))

    # Generate coordinates
    deltax = deltas*np.cos(theta_rad)
    deltay = deltas*np.sin(theta_rad)

    x = np.array([0]+list(np.cumsum(deltax)))
    y = np.array([0]+list(np.cumsum(deltay)))

    lmax = x[-1]
    ymax = np.max(np.abs(y))
    sinuosity = smax/lmax

    data = {'curve': curve, 'theta': theta, 's': s,
            'lmax': lmax, 'ymax': ymax, 'sinuosity': sinuosity}

    return x, y, data


def kinoshita_curve_zolezzi(theta_1: float, lambda_value: float, theta_s: float,
                    theta_f: float, n: int, m_points: int=1000,
                    ds: Union[None, float]=None):
    """
    Description:
    ------------

        Generate a Kinoshita Curve with the information related
        to the reach generated.
    ________________________________________________________________

    Args:
    ------------
    :param theta_0: float,
        Maximum angular amplitude in radians.
    :type theta_0: float
    :param lambda_value: float,
        Arc wavelength.
    :type lambda_value: float
    :param j_s: float,
        Skewness.
    :type j_s: float
    :param j_f: float,
        Flatness.
    :type j_f: float
    :param n: int,
        Number of loops.
    :type n: int
    :param m_points: int (default 1000),
        Number of points that describe the meander.
    :type m_points: int
    :param ds: float or None (default None),
        Delta of streamwise coordinate (s).
    :type ds: float or None
    :return:
        - x (numpy.ndarray) - X coordinates.
        - y (numpy.ndarray) - Y coordinates.
        - data (dict) - dict with 'curve': curvature, 'theta':
        values of theta in each iteration, 's': streamwise coordinates,
        'lmax': maximum length, 'ymax': maximum y extent,
        'sinuosity': sinuosity (sigma = smax/lmax).
    :rtype: (numpy.ndarray, numpy.ndarray, py:class:dict)
    """
    # Direction
    smax = n * lambda_value
    if ds is None:
        s = np.linspace(0, smax, m_points)
    else:
        s = np.arange(0, smax + ds, ds)
    deltas = s[1]

    k = 2*np.pi / lambda_value

    theta_rad = theta_1 * np.cos(k*s) + theta_s*np.sin(3*k*s) + theta_f*np.cos(3*k*s)

    curve = k*(theta_1*np.sin(k*s) - 3*theta_s*np.cos(3*k*s) + 3*theta_f*np.sin(3*k*s))

    theta = theta_rad*180/np.pi

    # Generate coordinates
    deltax = deltas*np.cos(theta_rad)
    deltay = deltas*np.sin(theta_rad)

    x = np.array([0]+list(np.cumsum(deltax)))
    y = np.array([0]+list(np.cumsum(deltay)))

    lmax = x[-1]
    ymax = np.max(np.abs(y))
    sinuosity = smax/lmax

    data = {'c': curve, 'theta': theta, 's': s,
            'lmax': lmax, 'ymax': ymax, 'sinuosity': sinuosity}

    return x, y, data


def rle(in_array):
    """
    Description:
    ------------

    Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy.

    ___________________________________________________________________________

    Args:
    ------------

    :param in_array:
        Array with values
    :type in_array: list
    :return:
        z: np.ndarray, run lengths.
        p: np.ndarray, start positions.
        ia: np.ndarray, values.
    """
    ia = np.asarray(in_array)                # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]                # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)    # must include last element posi
        z = np.diff(np.append(-1, i))        # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]


def calculate_curvature(ss, xs, ys, derivatives=None):
    """
    Description:
    ------------

        Calculate curvature from the coordinates and the length of
        the meander.

        It is based on Equation (2) in Vermeulen et al. (2016).

        Vermeulen, B., Hoitink, A. J. F., Zolezzi, G., Abad, J. D., & Aalto, R.
        (2016). Multiscale structure of meanders. Geophysical Research Letters,
        2016GL068238. https://doi.org/10.1002/2016GL068238
    ________________________________________________________________

    Args:
    ------------
    :param ss: np.ndarray
        Sreamwise coordinates
    :type ss: np.ndarray
    :param xs:
        X coordinates.
    :type xs: np.ndarray
    :param ys:
        Y coordinates.
    :type ys: np.ndarray
    :return:
        r: np.ndarray, Radius of curvature.
        c: np.ndarray, Curvature.
        s: np.ndarray, Streamwise coordinates.
    """
    if derivatives is None:
        dx = np.gradient(xs, ss)
        dy = np.gradient(ys, ss)
        d2x = np.gradient(dx, ss)
        d2y = np.gradient(dy, ss)
    else:
        dx = derivatives['dxds']
        dy = derivatives['dyds']
        d2x = derivatives['d2xds2']
        d2y = derivatives['d2yds2']
    c = (dx*d2y - dy*d2x) / (dx**2 + dy**2)**(3/2)
    c_r = copy.deepcopy(c)
    c_r[c_r == 0] = np.nan
    r = -1/c_r
    return r, c, ss


def get_inflection_points(c, s_curve):
    """

    Description:
    ------------

        Obtain the inflection points from the curvature.

    ________________________________________________________________

    Args:
    ------------
    :param c:
        Curvature.
    :type c: np.ndarray
    :param s_curve: np.ndarray,
        Streamwise coordinates with the same dimensions of c
    :type s_curve: np.ndarray
    :return:
        s_inf: np.ndarray, Streamwise inflection point.
        c_inf: np.ndarray, Curvature inflection point.
    """
    # Find inflexion points
    # condition_c = (c >= 0)
    condition_c = (c > 0)

    lengths, positions, type_run = rle(condition_c)
    
    ind_l = positions[1:] - 1
    ind_r = positions[1:]
    
    x_l = s_curve[ind_l]
    x_r = s_curve[ind_r]
    y_l = c[ind_l]
    y_r = c[ind_r]

    # Get Inflection points
    s_inf = -(x_r - x_l)/(y_r - y_l)*y_l + x_l
    c_inf = np.zeros_like(s_inf)
    return s_inf, c_inf, ind_l, ind_r


def calculate_direction_angle(ss, xs, ys, derivatives=None):
    """
    Description:
    ------------
        Calculates the direction angle along the planimetry coordinates.
    ____________________________________________________________________________

    Args:
    ------------
    :param dxds: np.ndarray,
        Derivative of x with respect to s.
    :type dxds: np.ndarray
    :param dyds: np.ndarray,
        Derivative of y with respect to s.
    :type dyds: np.ndarray
    :return:
        theta: np.ndarray, Direction angle.
    """
    if derivatives is None:
        dxds = np.gradient(xs, ss)
        dyds = np.gradient(ys, ss)
    else:
        dxds = derivatives['dxds']
        dyds = derivatives['dyds']
    # Theta 
    theta = np.arctan(dyds/dxds)
    # Condition 2
    cond = (dyds > 0) & (dxds < 0)
    theta[cond] = np.pi + theta[cond]
    # Condition 4
    cond = (dyds < 0) & (dxds < 0)
    theta[cond] = - np.pi + theta[cond]

    # Condition 3
    # cond = (dyds < 0) & (dxds > 0)
    # print(theta[cond][0])
    # theta[cond] = 2*np.pi + theta[cond]
    return theta


def translate(p, p1):
    """
    Description:
    ------------

        Translate points.

    ________________________________________________________________

    Args:
    ------------
    :param p: np.ndarray,
        Original coordinates as (n_points, n_variables)
    :type p: np.ndarray
    :param p1: np.ndarray
        Initial coordinates as (n_points, n_variables).
    :type p1: np.ndarray
    :return: np.ndarray,
        Translated points p.
    :rtype: np.ndarray
    """
    return p - p1


def rotate(p, p1, p2, theta=None):
    """
    Description:
    ------------

        Rotate points.

    ________________________________________________________________

    Args:
    ------------
    :param p: np.ndarray,
        Original coordinates as (n_points, n_variables)
    :type p: np.ndarray
    :param p1: np.ndarray
        Initial coordinates as (n_points, n_variables).
    :type p1: np.ndarray
    :param p2: np.ndarray,
        Ending coordinates as (n_points, n_variables).
    :type p2: np.ndarray
    :param theta: float (default, None),
        Rotating angle in radians. If None, the code will calculate
        the angle from p1 and p2
    :type theta: float
    :return: (np.ndarray, float),
        rotation_matrix: np.ndarray, Rotated points.
        theta: float, Angle of rotation.
    """
    if theta is None:
        # p2p1 = p2 - p1
        # p2p1_norm = np.sqrt(p2p1[0] ** 2 + p2p1[1] ** 2)
        # theta = np.arcsin((np.dot(p2 - p1, np.array([0, 1]))) / p2p1_norm)
        delta_x = p1[0] - p2[0]
        delta_y = p1[-1] - p2[-1]
        theta = np.arctan(delta_y / delta_x)
        while theta < 0.0:
            theta += np.pi * 2

    rotation_matrix = np.array(
        [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
    )

    if len(p.shape) > 1:
        return (rotation_matrix @ p.T).T, theta
    else:
        return rotation_matrix @ p, theta


def translate_rotate(points, index_initial, index_final, theta=None):
    """

    Description:
    ------------

        Translate and rotate points.

    ________________________________________________________________

    Args:
    ------------
    :param points: np.ndarray
         Original coordinates as (n_points, n_variables)
    :type points: np.ndarray
    :param index_initial: int,
         Index of initial coordinates
    :type index_initial: int,
    :param index_final: int,
         Index of final coordinates
    :type index_final: int
    :param theta: float,
         Rotating angle in radians. If None, the code will calculate
         the angle from p1 and p2
    :type theta: float
    :return:
        rotation_matrix: np.ndarray, Translated and rotated points.
        theta: float, Angle of rotation.
    """
    p1 = points[index_initial, :]
    p2 = points[index_final, :]

    translated_points = translate(points, p1)
    rotated_points, theta = rotate(translated_points, p1, p2, theta=theta)

    return rotated_points, theta


def get_reach_distances(x_coord):
    """
    Description:
    ------------

        This function calculates the distance from the starting point
        to the end points.
    ____________________________________________________________________________

    Args:
    ------------
    :param x_coord: np.ndarray,
        [x, y] coordinates in (n_points, 2)
    :type x_coord: np.ndarray
    :return: np.ndarray,
        Distance from the start point to the end point.
    """
    s_diff = np.diff(x_coord, axis=0)
    s_dist = np.sqrt((s_diff ** 2).sum(axis=1))
    s_dist = np.hstack((0, s_dist))
    s = np.cumsum(s_dist)
    return s


def fit_splines(s, x, y, method='geometric_mean', ds=0, k=3,
                smooth=0, ext=0, return_derivatives=True):
    """
    Description:
    ------------

        This function fits a spline to the coordinates with the
        minimum distance of the river.
    ________________________________________________________________

    Args:
    ------------
    :param s: numpy.ndarray,
        Distance vector
    :param x: numpy.ndarray,
        x coordiantes.
    :param y: numpy.ndarray,
        y coordiantes.
    :param method: str,
        Method to calculate the distance between points.
        Options: 'min', 'geometric_mean'
    :param ds: float,
        Distance between points. Default is None.
        If giving this parameter will override the method parameter.
    :param k: int,
        Order of the spline.
    :param smooth: float,
        Smoothness parameter.
    :param ext: int,
        Extrapolation method.
    :return:
        s_poly: numpy.ndarray, new distance vector.
        x_poly: numpy.ndarray, new x coordinates.
        y_poly: numpy.ndarray, new y coordinates.
    """
    method = method.lower()
    # ------------------
    # Fit the splines
    # ------------------
    if method == 'min' or method == 'minimum':
        diff_s = np.min(np.diff(s))
    elif method == 'geometric_mean':
        diff_s = 10 ** np.mean(np.log10(np.diff(s)))
    elif method == 'mean':
        diff_s = np.mean(np.diff(s))
    else:
        raise ValueError(f"method '{method} not implemented."
                         f"Please use 'min' or 'geometric_mean'")
    if ds > 0:
        diff_s = ds

    s_poly = np.arange(s[0], s[-1] + diff_s/2, diff_s)
    # ------------------
    # Generate Splines
    # -----------------
    x_spl = UnivariateSpline(s, x, k=k, s=smooth, ext=ext)
    y_spl = UnivariateSpline(s, y, k=k, s=smooth, ext=ext)
    x_poly = x_spl(s_poly)
    y_poly = y_spl(s_poly)
    splines = {'x_spl': x_spl, 'y_spl': y_spl}

    if return_derivatives:
        dxds = x_spl.derivative(n=1)(s_poly)
        dyds = y_spl.derivative(n=1)(s_poly)
        d2xds2 = x_spl.derivative(n=2)(s_poly)
        d2yds2 = y_spl.derivative(n=2)(s_poly)
        derivatives = {'dxds': dxds, 'dyds': dyds, 'd2xds2': d2xds2,
                       'd2yds2': d2yds2}
        return s_poly, x_poly, y_poly, derivatives, splines
    else:
        return s_poly, x_poly, y_poly


def fit_splines_complete(data, method='geometric_mean', ds=0,
                         k=3, smooth=0, ext=0):
    """
    Description:
    ------------
        Fit splines to all of the variables in the River class.
    ____________________________________________________________________________

    Args:
    ------------
    :param data: dict,
        Dictionary with the data. The dictionary must include the following
        's': incremental distance of stream
        'x': x coordinates of the river
        'y': y coordinates of the river
        'z': elevation of the river
        'da_sqkm': drainage area in square kilometers
        'w_m': width of the river
        'so': stream order
    :param method: str,
        Method to calculate the distance between points.
        Options: 'min', 'geometric_mean', 'mean', and 'width_based'
    :return:
        data: dict,
            Dictionary with the data. The dictionary must include the following
            's_poly': incremental distance of stream in the spline fit.
            'x_poly': x coordinates of the river in the spline fit.
            'y_poly': y coordinates of the river in the spline fit.
            'z_poly': elevation of the river in the spline fit.
            'da_sqkm_poly': drainage area in square kilometers
                            in the spline fit.
            'w_m_poly': width of the river in the spline fit.
            'so_poly': stream order in the spline fit.
    """
    # Extract data
    comid = np.array(data['comid'])
    so = data['so']
    s = data['s']
    x = data['x']
    y = data['y']
    z = data['z']
    da = data['da_sqkm']
    w = data['w_m']
    # Set smooth relative to the length of the data
    smooth = smooth * len(s)
    # ----------------------------------------
    # Calculate geometric meand of the width
    # ----------------------------------------
    # w_gm = 10 ** np.mean(np.log10(w))
    w_value = np.nanmin(w)
    if method == 'width_based' and not(np.isnan(w_value)):
        method = 'geometric_mean'
        ds = w_value
    elif method == 'width_based' and np.isnan(w_value):
        raise ValueError('The width value is NaN')
    # -------------------
    # Get coordinate poly
    # -------------------
    s_poly, x_poly, y_poly, derivatives, splines = fit_splines(
        s, x, y, method=method, ds=ds, k=k, smooth=smooth, ext=ext,
        return_derivatives=True)
    # ----------------------------------------
    # Generate Splines on the rest of the data
    # ----------------------------------------
    # x_spl = UnivariateSpline(s, x, k=k, s=smooth, ext=ext)
    # y_spl = UnivariateSpline(s, y, k=k, s=smooth, ext=ext)

    z_spl = UnivariateSpline(s, z, k=1, s=0, ext=0)
    f_comid = interpolate.interp1d(s, comid, fill_value=(comid[0], comid[-1]),
                                   kind='previous', bounds_error=False)
    f_so = interpolate.interp1d(s, so, fill_value=(so[0], so[-1]),
                                kind='previous', bounds_error=False)
    f_da = interpolate.interp1d(s, da, fill_value='extrapolate')
    f_w = interpolate.interp1d(s, w, fill_value='extrapolate')
    splines.update({'z_spl': z_spl, 'f_comid': f_comid, 'f_so': f_so,
                    'f_da': f_da, 'f_w': f_w})
    # ------------------
    # Create points
    # -----------------
    # x_poly = x_spl(s_poly)
    # y_poly = y_spl(s_poly)

    z_poly = z_spl(s_poly)
    comid_poly = f_comid(s_poly)
    so_poly = f_so(s_poly)
    da_poly = f_da(s_poly)
    w_poly = f_w(s_poly)
    # ------------------
    # Create data
    # -----------------
    data_fitted = {
        's_poly': s_poly ,'x_poly': x_poly, 'y_poly': y_poly,
        'z_poly': z_poly, 'comid_poly': comid_poly,
        'so_poly': so_poly, 'da_sqkm_poly': da_poly, 'w_m_poly': w_poly,
        'derivatives': derivatives, 'splines': splines}

    return data_fitted


def smooth_data(x, y, s, poly_order=2, savgol_window=2, gaussian_window=1):
    """
    Description:
    ------------
        Smooth the data using SavGol and Gaussian filters.
    ____________________________________________________________________________

    Args:
    ------------
    :param x: np.ndarray,
        x coordinates.
    :param y: np.ndarray,
        y coordinates.
    :param s: np.ndarray,
        Distance vector.
    :param poly_order: int,
        Order of the polynomial.
    :param savgol_window: int,
        Window size for the SavGol filter. Has to be an odd number.
    :param gaussian_window: int,
        Window size for the Gaussian filter.
    :return:
        s_smooth: np.ndarray, Smoothed distance vector.
        x_smooth: np.ndarray, Smoothed x coordinates.
        y_smooth: np.ndarray, Smoothed y coordinates.
    """
    # --------------------------
    # Extract data
    # --------------------------
    # Calculate ds
    ds = np.diff(s)[0]
    # --------------------------
    # Define Gaussian Kernel
    # --------------------------
    sigma = gaussian_window / 6
    t = np.linspace(-2.7 * sigma, 2.7 * sigma, gaussian_window)
    kernel = GF.gaussian_function(t, sigma)
    kernel /= np.sum(kernel)
    # --------------------------
    # Perform SavGol Filter
    # --------------------------
    if not savgol_window % 2:
        savgol_window += 1
    x_smooth = GF.savgol_filter(x, ds, poly_order, savgol_window, kernel)
    y_smooth = GF.savgol_filter(y, ds, poly_order, savgol_window, kernel)
    # Recalculate distance
    coords = np.vstack((x, y)).T
    s_smooth = get_reach_distances(coords)

    # Correct ds
    s_new_smooth = np.linspace(s_smooth[0], s_smooth[-1], len(s))
    ds_new = np.diff(s_new_smooth)[0]

    # refit splines
    s_smooth, x_smooth, y_smooth = fit_splines(s_smooth, x_smooth, y_smooth,
                                               ds=ds_new, return_derivatives=False)
    
    if len(x_smooth) != len(x):
        raise ValueError('The length of the smoothed data is different')

    return s_smooth, x_smooth, y_smooth


def calculate_lambda(x, y):
    """
    Description:
    ------------
        Calculate arc length of the transect.
    ____________________________________________________________________________

    Args:
    ------------
    :param x: np.ndarray,
        x coordinates.
    :param y: np.ndarray,
        y coordinates.
    :return:
        lambda: np.ndarray, Arc length of the transect.
    """
    coords = np.vstack((x, y)).T
    s_calc = get_reach_distances(coords)
    return s_calc[-1]


def calculate_l(x, y):
    """
    Description:
    ------------
        Calculate the length of the transect.
    ____________________________________________________________________________

    Args:
    ------------
    :param x: np.ndarray,
        x coordinates.
    :param y: np.ndarray,
        y coordinates.
    :return:
        l: np.ndarray, Length of the transect.
    """
    l = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
    return l


def calculate_sinuosity(l, lambda_value):
    """
    Description:
    ------------
        Calculate the sinuosity of the transect.
    ____________________________________________________________________________

    Args:
    ------------
    :param l: np.ndarray,
        Length of the transect.
    :param lambda_value: np.ndarray,
        Arc length of the transect.
    :return:
        sinuosity: np.ndarray, Sinuosity of the transect.
    """
    # Check valley distance
    if l == 0:
        sinuosity = np.nan
    else:
        sinuosity = lambda_value / l
    return sinuosity


def calculate_radius_of_curvature(x_st, y_st, x_end, y_end, x_mid, y_mid):
    """
    Description:
    ------------
        Calculate the radius of curvature of the transect.
    ____________________________________________________________________________

    Args:
    ------------
    """

    triangle = np.array([[x_st, y_st], [x_mid, y_mid], [x_end, y_end],
                         [x_st, y_st]]) 
    tri = Delaunay(triangle)
    cc = WTFunc.circumcenter(tri)
    x_c = cc[0]
    y_c = cc[1]
    radius = euclidean([x_mid, y_mid], [x_c, y_c])
    return x_c, y_c, radius


def calculate_assymetry(x, y, c):
    """
    Description:
    ------------
        Calculate the assymetry of the transect using Eq. 24 in Howard and
        Hemberger (1991).

        If the value is lower than zero the meander has an assymetry to the
        left, and if the value is higher than zero the meander has an
        assymetry to the right. For most NHDPlus information cases
        left is upstream and right is downstream.
    ____________________________________________________________________________

    Args:
    ------------
    :param x: np.ndarray,
        x coordinates.
    :param y: np.ndarray,
        y coordinates.
    :param c: np.ndarray,
        Curvature.
    :return: assymetry: float,
        Assymetry of the transect.
    """

    # Detect maximum point of curvature
    argmax_c = np.argmax(np.abs(c))
    # Calculate distances
    lambda_h = calculate_lambda(x, y)
    lambda_u = calculate_lambda(x[:argmax_c+1], y[:argmax_c+1])
    lambda_d = calculate_lambda(x[argmax_c:], y[argmax_c:])

    # Calculate assymetry
    a_h = (lambda_u - lambda_d)/lambda_h
    return a_h, lambda_h, lambda_u, lambda_d


def extend_node_bound(node, c):
    """
    Description:
    ------------
        Extend the bounds of a node in the meanders.
    ____________________________________________________________________________

    Args:
    ------------
    :param node: anytree node
        Node of the meanders.
    :param c: np.ndarray,
        Curvature of the transect.
    :return:
        node: anytree node, node with idx_planimetry_extended_start
              and idx_planimetry_extended_end.
    """
    # ------------------------------
    # Extract Information
    # ------------------------------
    idx_start = node.idx_planimetry_start
    idx_end = node.idx_planimetry_end
    c_meander = c[idx_start:idx_end + 1]
    # ------------------------------
    # Set extend values
    # ------------------------------
    idx_dif = np.abs(idx_end - idx_start)
    # Find side un curvature
    max_peak = np.max(c_meander)
    min_peak = np.abs(np.min(c_meander))
    if max_peak > min_peak:
        mult = -1
    else:
        mult = 1
    
    # get maximum differences in curvature inside the meander
    dif_c = np.abs(max_peak - min_peak)
    
    # ----------------------------------------------------
    # Find peaks to the left and right of the curvature
    # ----------------------------------------------------
    # Left
    peak_left = []
    idx_dif_left = copy.deepcopy(idx_dif)
    i = 0
    while len(peak_left) == 0:
        idx_left = idx_start - idx_dif_left
        if idx_left < 0:
            idx_left = 0
        val_range_left = np.arange(idx_left, idx_start+1, 1).astype(int)
        c_left = mult*c[val_range_left]
        peak_left, _ = find_peaks(c_left)
        # Selected Value with highest curvature
        if len(peak_left) > 0:
            # Find Peaks
            c_at_peaks_left = c_left[peak_left]
            max_c_left = np.max(c_at_peaks_left)
            closer_c_left = c_at_peaks_left[-1]
            dif_c_left = np.abs(max_c_left - closer_c_left)
            idx_peak_left = val_range_left[c_left == max_c_left][0]
            # # Compare values to pick the best curvature peak
            # if dif_c_left >= 0.2*dif_c:
            #     idx_peak_left = val_range_left[c_left == max_c_left][0]
            # else:
            #     idx_peak_left = val_range_left[peak_left[-1]]
        else:
            idx_peak_left = copy.deepcopy(idx_start)
            idx_dif_left += idx_dif//2
        i += 1
        if i > 10:
            break

    # Right
    peak_right = []
    idx_dif_right = copy.deepcopy(idx_dif)
    i = 0
    while len(peak_right) == 0:
        idx_right = idx_end + idx_dif_right
        if idx_right >= len(c):
            idx_right = len(c) - 1
        val_range_right = np.arange(idx_end, idx_right + 1, 1).astype(int)
        c_right = mult*c[val_range_right]
        peak_right, _ = find_peaks(c_right)
        # Selected Value with highest curvature
        if len(peak_right) > 0:
            # Find Peaks
            c_at_peaks_right = c_right[peak_right]
            max_c_right = np.max(c_at_peaks_right)
            closer_c_left = c_at_peaks_right[0]
            dif_c_right = np.abs(max_c_right - closer_c_left)
            # print(c_at_peaks_right)
            # print(dif_c_right, 0.1*dif_c)
            idx_peak_right = val_range_right[c_right == max_c_right][0]
            # Compare values to pick the best curvature peak
            # if dif_c_right >= 0.2*dif_c:
            #     idx_peak_right = val_range_right[c_right == max_c_right][0]
            # else:
            #     idx_peak_right = val_range_right[peak_right[0]]
        else:
            idx_peak_right = copy.deepcopy(idx_end)
            idx_dif_right += idx_dif//2
        i += 1
        if i > 10:
            break
    

    node.idx_planimetry_extended_start = idx_peak_left
    node.idx_planimetry_extended_end = idx_peak_right

    return node

@DeprecationWarning
def extend_bounds(bounds, c, x, y, meander_id=None, extend=0, sk=None, fl=None):
    """
    Description:
    ------------
        Extend the bounds of the meanders.
    ____________________________________________________________________________

    Args:
    ------------
    :param bounds: np.ndarray,
        Bounds of the meanders.
    :param c: np.ndarray,
        Curvature of the transect.
    :param meander_id: np.ndarray,
        Meander id of the bounds.
    :param extend: int,
        Number of points to extend the bounds after maximum points on each side.
    :param sk: np.ndarray,
        Skewness of the transect meanders.
    :param fl: np.ndarray,
        Flatness of the transect meanders.
    :return:
        new_bounds: np.ndarray, Extended bounds of the meanders.
        ind_curvature: np.ndarray, Indicator if up or down in curvature.
    """

    # Set new bounds based on peaks in the curvature
    new_bounds = np.zeros((bounds.shape[0], 2), dtype=int)
    # Indicator if up or down in curvature
    ind_curvature = np.zeros(bounds.shape[0])
    for id_m in range(0, len(bounds)):
        start_meander = bounds[id_m, 0]
        end_meander = bounds[id_m, 1]
        c_meander = c[start_meander:end_meander + 1]
        x_m = x[start_meander:end_meander + 1]
        y_m = y[start_meander:end_meander + 1]
        # --------------------
        # Extend meander
        # --------------------
        # Find peaks in the curvature of the bounds
        dif = np.abs(end_meander - start_meander)//2
        # Set minimum value of points to look for peaks
        # if dif < 4:
        #     dif = 4
        # --------------------
        # Find Peaks
        # --------------------
        max_peak = np.max(c_meander)
        min_peak = np.abs(np.min(c_meander))
        # if np.mean(c_meander) > 0:
        # if np.median(c_meander) > 0:
        # if np.sum(c_meander > 0)/len(c_meander) >= 0.51:
        if max_peak > min_peak:
            mult = -1
        else:
            mult = 1

        # Correct depending on skewness and sinuosity
        mult_right = copy.deepcopy(mult)
        mult_left = copy.deepcopy(mult)
        dif_right = copy.deepcopy(dif)
        dif_left = copy.deepcopy(dif)
        l_value = calculate_l(x_m, y_m)
        lambda_value = calculate_lambda(x_m, y_m)
        sn = lambda_value / l_value

        # if sk is not None:
        #     if np.abs(sk[id_m]) > 1e-3 and sn > 1.5:
        #         if sk[id_m] > 0:
        #             mult_right = -1 * mult
        #             # dif_right += dif
        #         if sk[id_m] < 0:
        #             mult_left = -1 * mult
        #             # dif_left += dif

        # Look for peaks in curvature
        peak_left = []
        peak_right = []
        dif_update = copy.deepcopy(dif)
        iter = 0
        while len(peak_left) == 0 or len(peak_right) == 0:
            ind_left = (start_meander - dif_left - extend)
            if ind_left < 0:
                ind_left = 0
            val_range = np.arange(ind_left, start_meander + 2, 1).astype(int)
            peak_left = find_peaks(mult_left * c[val_range])[0]

            ind_right = (end_meander + dif_right + extend)
            if ind_right >= len(c):
                ind_right = len(c) - 1
            val_range = np.arange(end_meander - 2, ind_right + 1, 1).astype(int)
            peak_right = find_peaks(mult_right * c[val_range])[0]
            if len(peak_left) == 0 or len(peak_right) == 0:
                dif_update += dif
                dif_right += dif
                dif_left += dif
            if iter > 5:
                if len(peak_left) == 0:
                    peak_left = np.array([start_meander])
                if len(peak_right) == 0:
                    peak_right = np.array([end_meander])
                break
            iter += 1

        ind_left = peak_left + start_meander - dif_left - extend
        peak_left = peak_left[ind_left >= extend]
        ind_left = ind_left[ind_left >= extend]
        if np.all(peak_left == start_meander) or len(ind_left) == 0:
            new_start = start_meander
        else:
            c_p_left = c[ind_left]
            ind_left = np.argmax(np.abs(c_p_left))
            new_start = peak_left[ind_left] + start_meander - dif_left -\
                        extend

        ind_right = peak_right + end_meander + extend
        if np.any(ind_right >= len(c)):
            cond = ind_right >= len(c)
            ind_right[cond] = [len(c) - 1 for _ in range(np.sum(cond))]
        if np.all(peak_right == end_meander):
            new_end = end_meander
        else:
            c_p_right = c[ind_right]
            ind_right = np.argmax(np.abs(c_p_right))
            new_end = peak_right[ind_right] + end_meander + extend

        # Store Information
        new_bounds[id_m, :] = [new_start, new_end]
        ind_curvature[id_m] = mult

    # Clean bounds
    if sk is None:
        sk = np.array([np.nan for _ in range(len(new_bounds))])
    if fl is None:
        fl = np.array([np.nan for _ in range(len(new_bounds))])

    pos_values = np.where(ind_curvature == 1)[0]
    bounds_pos = new_bounds[pos_values, :]
    ind_pos = ind_curvature[pos_values]
    meander_id_pos = meander_id[pos_values]
    sk_pos = sk[pos_values]
    fl_pos = fl[pos_values]

    new_bounds_pos, new_ind_pos, new_meander_id_pos, new_sk_pos, new_fl_pos =\
        remove_duplicates(bounds_pos, ind_pos, meander_id_pos, sk_pos, fl_pos)

    neg_values = np.where(ind_curvature == -1)[0]
    bounds_neg = new_bounds[neg_values, :]
    ind_neg = ind_curvature[neg_values]
    meander_id_neg = meander_id[neg_values]
    sk_neg = sk[neg_values]
    fl_neg = fl[neg_values]

    new_bounds_neg, new_ind_neg, new_meander_id_neg, new_sk_neg, new_fl_neg =\
        remove_duplicates(bounds_neg, ind_neg, meander_id_neg, sk_neg, fl_neg)

    if len(new_bounds_pos) == 0:
        new_bounds_corr = new_bounds_neg
    elif len(new_bounds_neg) == 0:
        new_bounds_corr = new_bounds_pos
    else:
        new_bounds_corr = np.vstack((new_bounds_pos, new_bounds_neg))

    ind_curvature_corr = np.hstack((new_ind_pos, new_ind_neg))
    meander_id_corr = np.hstack((new_meander_id_pos, new_meander_id_neg))
    new_sk_corr = np.hstack((new_sk_pos, new_sk_neg))
    new_fl_corr = np.hstack((new_fl_pos, new_fl_neg))

    return new_bounds_corr, ind_curvature_corr, meander_id_corr, new_sk_corr, \
        new_fl_corr


def remove_duplicates(bounds, ind_curvature, meander_id, sk, fl):
    """
    Description:
    ------------
        Remove duplicate bounds.
    ____________________________________________________________________________

    Args:
    ------------
    :param bounds: np.ndarray,
        Bounds of the meanders.
    :param ind_curvature: np.ndarray,
        Indicator if up or down in curvature.
    :param meander_id: np.ndarray,
        Meander id.
    :param sk: np.ndarray,
        Skewness.
    :param fl: np.ndarray,
        Flatness.
    :return:
        bounds: np.ndarray, Bounds of the meanders.
        ind_curvature: np.ndarray, Indicator if up or down in curvature.
    """

    # Remove duplicates
    new_bounds = []
    ind_curvature_new = []
    new_meander_id = []
    new_sk = []
    new_fl = []
    # Remove Start
    starts = bounds[:, 0]
    ends = bounds[:, 1]
    duplicate_values_start = np.where(np.diff(starts) == 0)[0]
    duplicate_values_end = np.where(np.diff(ends) == 0)[0]
    remove = []

    for i_b, bound in enumerate(bounds):
        if i_b in duplicate_values_start:
            bound_d = bounds[i_b: i_b + 2, :]
            new_end = np.max(bound_d[:, 1])
            bound = [bound_d[0, 0], new_end]
        elif i_b in duplicate_values_start + 1:
            continue

        if i_b in duplicate_values_end:
            bound_d = bounds[i_b: i_b + 2, :]
            bound = [bound_d[0, 0], bound_d[1, 1]]
        elif i_b in duplicate_values_end + 1:
            continue

        new_bounds.append(bound)
        ind_curvature_new.append(ind_curvature[i_b])
        new_meander_id.append(meander_id[i_b])
        new_sk.append(sk[i_b])
        new_fl.append(fl[i_b])

    # Clean Database of redundant meanders
    new_bounds = np.array(new_bounds)
    new_bounds_2 = []
    ind_curvature_new_2 = []
    new_meander_id_2 = []
    new_sk_2 = []
    new_fl_2 = []
    for i_b, bound in enumerate(new_bounds):
        bound = new_bounds[i_b, :]
        array = np.arange(bound[0], bound[1] + 1)
        intercept_start = np.intersect1d(array, new_bounds[:, 0])
        intercept_end = np.intersect1d(array, new_bounds[:, 1])
        if len(intercept_start) > 1:
            ind_start = []
            for intercept_start_i in intercept_start:
                ind_1 = np.where(new_bounds[:, 0] == intercept_start_i)[0]
                for i in ind_1:
                    ind_start.append(int(i))
            ind_start = np.array(ind_start)
            ind_start = ind_start[ind_start != i_b]
            if type(ind_start) in (int, np.int64, np.int32, float):
                ind_start = [ind_start]
        else:
            ind_start = []
        
        if len(intercept_end) > 1:
            ind_end = []
            for intercept_end_i in intercept_end:
                ind_1 = np.where(new_bounds[:, 1] == intercept_end_i)[0]
                for i in ind_1:
                    ind_end.append(i)
            ind_end = np.array(ind_end)
            ind_end = ind_end[ind_end != i_b]
            if isinstance(ind_end, int):
                ind_end = [ind_end]
        else:
            ind_end = []
        
        if len(ind_start) > 0 and len(ind_end) > 0:
            for i_s in ind_start:
                for i_e in ind_end:
                    if i_s == i_e:
                        remove.append(i_s)
        
        if i_b in remove:
            continue

        new_bounds_2.append(bound)
        ind_curvature_new_2.append(ind_curvature_new[i_b])
        new_meander_id_2.append(new_meander_id[i_b])
        new_sk_2.append(new_sk[i_b])
        new_fl_2.append(new_fl[i_b])

    new_bounds = np.array(new_bounds_2)
    ind_curvature_new = np.array(ind_curvature_new_2)
    new_meander_id = np.array(new_meander_id_2)
    new_sk = np.array(new_sk_2)
    new_fl = np.array(new_fl_2)

    return new_bounds, ind_curvature_new, new_meander_id, new_sk, new_fl


@DeprecationWarning
def aggregate_bounds(bounds, x, y, ind_curvature, value_mult=1,
                     sinuosity_threshold=1.05):
    """
    Description:
    ------------
        Aggregate bounds based on the length of meanders and curvature
    ____________________________________________________________________________

    Args:
    ------------
    :param bounds: np.ndarray,
        Bounds of the meanders.
    :param x: np.ndarray,
        River x coordinates.
    :param y: np.ndarray,
        River y coordinates.
    :param ind_curvature: np.ndarray,
        Indicator if up or down in curvature.
    :param value_mult: int,
        Indicator if up or down in curvature. up=1, down=-1.
    :param sinuosity_threshold: float,
        Threshold for the sinuosity.
    :return:
        bounds: np.ndarray, Aggregated bounds of the meanders.
    """

    # Get the bounds on one side of the curvature
    one_side_curvature = np.where(ind_curvature == value_mult)[0]
    bounds_one_side = bounds[one_side_curvature, :]
    aggr_bounds = []
    sinuosity_all = []
    # Loop through one side of the curvature
    i = 0
    while i < len(one_side_curvature):
        # index = one_side_curvature[i]
        start = copy.deepcopy(bounds_one_side[i, 0])
        end = copy.deepcopy(bounds_one_side[i, 1])
        initial_x_m = x[start: end]
        initial_y_m = y[start: end]

        # Calculate horizontal length
        l_value = [calculate_l(initial_x_m, initial_y_m)]

        # Calculate reach length
        lamba_value = [calculate_lambda(initial_x_m, initial_y_m)]

        # Calculate sinuosity
        sinuosity = [calculate_sinuosity(l_value[-1], lamba_value[-1])]

        if i >= len(one_side_curvature):
            break
        for j in range(1, 6):
            if i + j >= len(one_side_curvature):
                break
            new_end = bounds_one_side[i + j, 1]
            x_m = x[start: new_end]
            y_m = y[start: new_end]
            # Calculate horizontal length
            l_value.append(calculate_l(x_m, y_m))
            # Calculate reach length
            lamba_value.append(calculate_lambda(x_m, y_m))
            # Calculate sinuosity
            sinuosity.append(calculate_sinuosity(l_value[-1], lamba_value[-1]))

        # Extract meander with the highest sinuosity
        # Check for minimum horizontal distance
        if len(l_value) == 1:
            j_index = 0
        else:
            min_arg_l_1 = np.argmin(l_value)
            min_arg_l_2 = np.argmin(l_value[1:]) + 1
            if min_arg_l_2 - min_arg_l_1 == 1:
                j_index = min_arg_l_1
            else:
                j_index = min_arg_l_2

        # Check for sinuosity
        if sinuosity[j_index] < sinuosity_threshold:
            i += 1
            continue

        # Store Information
        new_end = bounds_one_side[i + j_index, 1]
        sinuosity_all.append(sinuosity[j_index])
        aggr_bounds.append([start, new_end])
        if j_index == 0:
            i += 1
        else:
            i += j_index + 1

    return np.array(aggr_bounds)


def calculate_coordinates_from_curvature(s_curvature, c, x, y):
    initial_coords = np.array([x[0], y[0]])
    known_point = np.array([x[1], y[1]])
    segments_length = np.diff(s_curvature)

    x_r = [initial_coords[0]]
    y_r = [initial_coords[1]]
    current_pos = np.copy(known_point)
    initial_direction = known_point - initial_coords
    initial_direction /= np.linalg.norm(initial_direction)
    current_direction = np.copy(initial_direction)

    for i, c_v in enumerate(c[1:]):
        # Estimate arc length from segment length
        arc_length = segments_length[i]
        # Estimate change in angle along the arc
        delta_theta = c_v * arc_length
        # Update direction vector using polar coordinates
        current_direction = np.dot(
            np.array([[np.cos(delta_theta), -np.sin(delta_theta)],
                      [np.sin(delta_theta), np.cos(delta_theta)]]),
            current_direction)
        # Update x and y coordinates using direction vector and arc length
        delta_x = arc_length * current_direction[0]
        delta_y = arc_length * current_direction[1]
        current_pos[0] += delta_x
        current_pos[1] += delta_y
        x_r.append(current_pos[0])
        y_r.append(current_pos[1])

    return np.array(x_r), np.array(y_r)


def calculate_channel_width(da):
    """
    Description
    -------------

    Calculate the channel width from the drainage area.

    This function uses equation (15) presented in Wilkerson et al. (2014).
    
    Wilkerson, G. V., Kandel, D. R., Perg, L. A., Dietrich, W. E., Wilcock,
    P. R., & Whiles, M. R. (2014). Continental-scale relationship between
    bankfull width and drainage area for single-thread alluvial channels.
    Water Resources Research, 50. https://doi.org/10.1002/2013WR013916

    ____________________________________________________________________________

    Args:
    ------------
    :param da: np.ndarray,
        Drainage area in km^2.
    :return:
        w: np.ndarray,
            Channel width.
    """

    if isinstance(da, int) or isinstance(da, float):
        da = np.array([da])
    
    cond_1 = (np.log(da) < 1.6)
    cond_2 = (np.log(da) >= 1.6) & (np.log(da) < 5.820)
    cond_3 = (np.log(da) >= 5.820)

    w = np.zeros_like(da)

    w[cond_1] = 2.18*da[cond_1]**0.191
    w[cond_2] = 1.41*da[cond_2]**0.462
    w[cond_3] = 7.18*da[cond_3]**0.183

    # Threshold 
    w[w < 1] = 1

    return w


def calculate_spectrum_cuts(s, c):
    """
    
    """
    # wave = np.abs(wave**2)
    # wave_sum = np.sum(wave, axis=0)
    # peaks_max, _ = find_peaks(wave_sum, height=0)
    # max_wave = wave_sum[peaks_max]
    # max_s = s[peaks_max]
    # # peaks_min, _ = find_peaks(max_wave, prominence=np.std(max_wave))
    # peaks_min, _ = find_peaks(-max_wave, prominence=np.std(max_wave))

    # min_s = max_s[peaks_min]
    # import ruptures as rpt

    # # change point detection
    # print(max_wave.shape, max_s.shape)
    # algo = rpt.Pelt(model="rbf").fit(max_wave)
    # change_location1 = np.array(algo.predict(pen=0.5))

    # min_s = max_s[change_location1[change_location1 < len(max_s)]]

    # Use the Morlet Wavelet to perfom the separation
    ds = np.diff(s)[0]
    values = WTFunc.calculate_cwt(c, ds, mother='MORLET')
    wave = values[0]

    wave = np.abs(wave**2)
    wave_sum = np.sum(wave, axis=0)
    peaks_min, _ = find_peaks(-wave_sum, prominence=np.std(wave_sum))
    min_s = s[peaks_min]


    return peaks_min, min_s


def get_reach_from_network(hw, reach_generator,
                           min_distance, method, calculate_poly,
                           linking_network, comid_network,
                           loading_from_file, comid_network_file,
                           coords_file=None):
    """
    """
    # --------------------------
    # keys
    # --------------------------
    keys = ['s', 'x', 'y', 'z', 'comid', 'so',
            'da_sqkm', 'w_m']
    keys_lab = {i: i for i in keys}
    # Loading from file
    time1 = time.time()
    if loading_from_file:
        comid_network = FM.load_data(
            comid_network_file, keys=[str(hw)])
    comid_list = list(comid_network[str(hw)])
    # print('Loading Network')
    # utl.toc(time1)

    # --------------------------
    # Verify reach length
    # --------------------------
    lengthkm = reach_generator.data_info.loc[
        comid_list, 'lengthkm'].values
    total_length = np.sum(lengthkm)
    remove = 0
    i_rep = 0
    cut = 10
    while total_length*1000 < min_distance:
        additional_comid = linking_network.loc[comid_list[-1],
                                                'linking_comid']
        # print(i_rep, additional_comid)
        if additional_comid == 0 or i_rep >= cut or (
            additional_comid in comid_list):
            remove = 1
            break
        comid_list.append(additional_comid)
        lengthkm = reach_generator.data_info.loc[
            comid_list, 'lengthkm'].values
        total_length = np.sum(lengthkm)
        i_rep += 1
    # print('Extending reach')
    # utl.toc(time1)

    # --------------------------
    # Extract original coordinates
    # --------------------------
    time1 = time.time()
    data_pd = reach_generator.map_coordinates(
        comid_list, file_coords=coords_file)
    # print('Extracting original coordinates')
    # utl.toc(time1)
    # Calculate Distance
    distance = data_pd['s'].values[-1]
    # print(data_pd['s'].shape)
    delta_time_extract = time.time() - time1 
    # Remove that reach
    if remove == 1 or len(data_pd['x'].values) <= 3:
        remove = 0
        return {str(hw): {'start_comid': -1}}
            
    time1 = time.time()
    if calculate_poly:
        try:
            data = reach_generator.fit_splines(
                data_pd, method=method)
            keys_lab.update({i: f'{i}_poly' for i in keys})
        except:
            print(f'Error in reach {hw}')
            return {str(hw): {'start_comid': -1}}
        comid = data[keys_lab['comid']]
    else:
        comid = np.array(data_pd.index)
        data = {}
        for key in list(data_pd):
            data[key] = data_pd[key].values
    
    # print('Calculating Polynomial')
    # utl.toc(time1)
    delta_time_poly = time.time() - time1

    huc_n = linking_network.loc[hw, 'huc_n']

    # Save data into dict
    time1 = time.time()
    data = {str(hw): {
        key: data[keys_lab[key]] for key in keys}}
    # data[str(hw)]['huc04'] = self.huc04
    data[str(hw)]['huc_n'] = huc_n
    data[str(hw)]['start_comid'] = hw
    data[str(hw)]['time_extract'] = delta_time_extract
    data[str(hw)]['time_poly'] = delta_time_poly
    data[str(hw)]['time_poly'] = delta_time_poly
    # print('Storing Information')
    # utl.toc(time1)
    
    # data_to_save[str(hw)].update({
    #     key: data[keys_lab[key]] for key in keys})
    # data_to_save[str(hw)]['huc04'] = self.huc04
    # data_to_save[str(hw)]['huc_n'] = huc_n
    # data_to_save[str(hw)]['start_comid'] = hw
    # data_to_save[str(hw)]['uid'] = hw

    return data


