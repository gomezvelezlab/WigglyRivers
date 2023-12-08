# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2023-11-19
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
______________________________________________________________________________

 DESCRIPTION:
   Functions related to meander creation and fitting.
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
from circle_fit import taubinSVD

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


def kinoshita_curve_abad(theta_0: float, lambda_value: float, j_s: float,
                    j_f: float, n: int, m_points: int=1000,
                    ds: Union[None, float]=None):
    """
    Description:
    ------------

        Generate a Kinoshita Curve with the information related
        to the reach generated.

        The Kinoshita curve is based on (Kinoshita, 1961). The 
        equations presented in this function are based on the
        equations presented in (Abad and Garcia, 2009).

        References:
        ------------
        Abad, J. D., & Garcia, M. H. (2009). Experiments in a
        high-amplitude Kinoshita meandering channel: 1. Implications
        of bend orientation on mean and turbulent flow structure:
        KINOSHITA CHANNEL, 1. Water Resources Research, 45(2).
        https://doi.org/10.1029/2008WR007016

        Kinoshita, R. (1961). Investigation of channel
        deformation in Ishikari River. Report of Bureau of
        Resources, 174. Retrieved from
        https://cir.nii.ac.jp/crid/1571417124444824064
    ____________________________________________________________________________

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


def kinoshita_curve_zolezzi(theta_0: float, lambda_value: float, theta_s: float,
                    theta_f: float, n: int, m_points: int=1000,
                    ds: Union[None, float]=None):
    """
    Description:
    ------------

        Generate a Kinoshita Curve with the information related
        to the reach generated.

        The Kinoshita curve is based on (Kinoshita, 1961). The 
        equations presented in this function are based on the
        equations presented in (Zolezzi and Güneralp, 2016).

        References:
        ------------
        Kinoshita, R. (1961). Investigation of channel
        deformation in Ishikari River. Report of Bureau of
        Resources, 174. Retrieved from
        https://cir.nii.ac.jp/crid/1571417124444824064

        Zolezzi, G., & Güneralp, I. (2016). Continuous wavelet
        characterization of the wavelengths and regularity of
        meandering rivers. Geomorphology, 252, 98–111.
        https://doi.org/10.1016/j.geomorph.2015.07.029

    ____________________________________________________________________________

    Args:
    ------------
    :param theta_0: float,
        Maximum angular amplitude in radians.
    :type theta_0: float
    :param lambda_value: float,
        Arc wavelength.
    :type lambda_value: float
    :param theta_s: float,
        coefficient for Skewness in radians.
    :type theta_s: float
    :param theta_f: float,
        coefficient for Fatness in radians.
    :type theta_f: float
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

    theta_rad = theta_0 * np.cos(k*s) + theta_s*np.sin(3*k*s) + theta_f*np.cos(3*k*s)

    curve = k*(theta_0*np.sin(k*s) - 3*theta_s*np.cos(3*k*s) + 3*theta_f*np.sin(3*k*s))

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

        Calculate curvature and the direction angle from the coordinates and
        the arc-length of the river transect.

        The equation for curvature and direction angle are based on the
        equations presented in (Güneralp and Rhoads, 2008).

        If the derivatives are not provided, the function will calculate them
        using the np.gradient function.

        References:
        ------------
        Güneralp, İ., & Rhoads, B. L. (2008). Continuous Characterization of the
        Planform Geometry and Curvature of Meandering Rivers. Geographical
        Analysis, 40(1), 1–25. https://doi.org/10.1111/j.0016-7363.2007.00711.x
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
        - r: np.ndarray, Radius of curvature.
        - c: np.ndarray, Curvature.
        - theta: np.ndarray, direction angle
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

    # --------------------------------
    # Calculate direction angle
    # --------------------------------
    segm_length = np.diff(ss)
    theta = np.zeros_like(ss)
    # Start with known point
    theta[0] = np.arctan(dy/dx)[0]
    theta[1:] = c[1:] * segm_length
    theta = np.cumsum(theta)
    return r, c, theta


def get_inflection_points(s, c):
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
    
    x_l = s[ind_l]
    x_r = s[ind_r]
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
        Calculates the direction angle using the coordinates. Keep in mind that
        this calculation would not work if the river direction is in the second
        and third cartesian quadrants from the start of the river.

        To have a better estimate of the direction angle use the function
        RiverFunctions.calculate_curvature(...).
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
    
    # -------------------------------------------------------------------------
    # These computations have complications when the river is rotated. Making
    #   the angle jump between positive and negative angles once we complete
    #   a loop.

    # Direction-angle
    alpha = np.arctan(dyds/dxds)
    # alpha = np.arctan(dxds/-dyds)
    # Condition 1
    theta = copy.deepcopy(alpha)
    # Condition 2
    cond = (dyds > 0) & (dxds < 0)
    theta[cond] = np.pi + alpha[cond]
    # Condition 4
    cond = (dyds < 0) & (dxds < 0)
    theta[cond] = -np.pi + alpha[cond]

    return theta


def calculate_direction_azimuth(ss, xs, ys, derivatives=None):
    """
    Description:
    ------------
        Calculates the direction azimuth using the coordinates. Keep in mind
        that this calculation would not work if the river direction is in the
        second and third cartesian plane quadrants from the start of the river.

        To have a better estimate of the direction angle use the function
        RiverFunctions.calculate_curvature(...) and convert the angles to
        azimuth.
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
    
    # -------------------------------------------------------------------------
    # These computations have complications when the river is rotated. Making
    #   the angle jump between positive and negative angles once we complete
    #   a loop.
    # Azimuth
    # For cuadrant 1
    alpha = np.arctan(dyds/dxds)
    theta = np.pi/2 - alpha

    # For cuadrant 2
    cond = (dyds >= 0) & (dxds < 0)
    theta[cond] = 3*np.pi/2 - alpha[cond]

    # For cuadrant 3
    cond = (dyds < 0) & (dxds > 0)
    theta[cond] = np.pi/2 - alpha[cond]

    # For cuadrant 4
    cond = (dyds < 0) & (dxds < 0)
    theta[cond] = 3*np.pi/2 - alpha[cond]
    return theta


def translate(p, p1):
    """
    Description:
    ------------

        Translate points.
    ____________________________________________________________________________

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
    ____________________________________________________________________________

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
    ____________________________________________________________________________

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

        This function calculates the cummulative streamwise distance of the
        river transect using the coordinates.
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
    # # -------------------------
    # # Equally spaced values
    # # -------------------------
    # # Use approximation from Guneralp et al. (2008)
    # # Create vector with indices
    # tau = np.arange(0, len(s))
    # # Create splines for x and y with the indices
    # x_spl_tau = UnivariateSpline(tau, x, k=k, s=0, ext=ext)
    # y_spl_tau = UnivariateSpline(tau, y, k=k, s=0, ext=ext)
    # # Recalculate the position of the indices scaled to the actual distance
    # #  of each initial point
    # tau_new = s/s[-1] * tau[-1]

    # # Evaluate the splines in the new indices
    # x_new = x_spl_tau(tau_new)
    # y_new = y_spl_tau(tau_new)

    # # Create spline at the new scaled points and evaluate it on equally sampled
    # # points
    # s_scaled = np.linspace(np.min(tau_new), np.max(tau_new), len(s_poly))
    # x_spl_tau_new = UnivariateSpline(tau_new, x_new, k=k, s=0, ext=ext)
    # y_spl_tau_new = UnivariateSpline(tau_new, y_new, k=k, s=0, ext=ext)
    # x_reg = x_spl_tau_new(s_scaled)
    # y_reg = y_spl_tau_new(s_scaled)
    # # ------------------
    # # Generate Splines
    # # -----------------
    # x_spl = UnivariateSpline(s_poly, x_reg, k=k, s=smooth, ext=ext)
    # y_spl = UnivariateSpline(s_poly, y_reg, k=k, s=smooth, ext=ext)
    x_spl = UnivariateSpline(s, x, k=k, s=smooth, ext=ext)
    y_spl = UnivariateSpline(s, y, k=k, s=smooth, ext=ext)
    x_poly = x_spl(s_poly)
    y_poly = y_spl(s_poly)
    # s_poly_2 = get_reach_distances(np.vstack((x_poly, y_poly)).T)
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
        Calculate wavelength of the transect.
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
        Calculate the valley length of the transect.
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


def calculate_radius_of_curvature(x, y, wavelength):
    """
    Description:
    ------------
        Calculate the radius of curvature of the meander by fitting a circle
        to the half-meander section and using the wavelength as the arc length.
    ____________________________________________________________________________

    Args:
    ------------
    :param x: np.ndarray,
        x coordinates.
    :param y: np.ndarray,
        y coordinates.
    :param wavelength: float,
        Wavelength of the meander.
    :return:
        - x_c: float, x coordinate of the center of the circle.
        - y_c: float, y coordinate of the center of the circle.
        - radius: float, radius of the circle.
    """
    # Fit Circle
    coordinates = np.vstack((x, y)).T
    x_mid = x[len(x)//2]
    y_mid = y[len(y)//2]
    x_cen, y_cen, r, sigma = taubinSVD(coordinates)

    # Calculate Omega
    w = wavelength / (2 * np.pi)
    rvec = np.array([x_cen - x_mid, y_cen - y_mid])/r

    x_c = x_mid + rvec[0] * w
    y_c = y_mid + rvec[1] * w

    radius = np.sqrt((x_c - x_mid)**2 + (y_c - y_mid)**2)
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

        References:
        ------------
        Howard, A. D., & Hemberger, A. T. (1991). Multivariate characterization
        of meandering. Geomorphology, 4(3–4), 161–186.
        https://doi.org/10.1016/0169-555X(91)90002-R

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


def extend_meander_bound_database(meander_database):
    """
    Description:
    ------------
        Extend the bounds of the meanders in the database. The extension 
        depends on the maximum points in the curvature of the previous
        and next meanders.
    ____________________________________________________________________________

    """

    return meander_database


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
    # TODO: Change this function according to the maximum and minimum of the
    #  curvature of the adjacent meanders.
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
    
    References:
    ------------
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
    Description:
    -------------
        Calculate the spectrum cuts of the curvature.
    ____________________________________________________________________________

    Args:
    ------------
    :param s: np.ndarray,
        Streamwise distance vector.
    :param c: np.ndarray,
        Curvature.
    :return:
        - peaks_min: np.ndarray, Indices of the minima of the curvature.
        - min_s: np.ndarray, Streamwise distance of the minima of the curvature.
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
