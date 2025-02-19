# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2025-02-16
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
    Functions related to meander creation and fitting.
"""
# -----------
# Libraries
# -----------
from typing import Union, Tuple
import copy
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy.signal import find_peaks
from circle_fit import taubinSVD
from scipy.interpolate import splprep, splev
from anytree import Node

# Package packages
from ..utilities import general_functions as GF

# from ..utilities.classExceptions import *
from ..wavelet_tree import WaveletTreeFunctions as WTFunc


# -----------
# Functions
# -----------
def convert_str_float_list_vector(x_val: str) -> np.ndarray:
    """Convert string to float vector

    example:

    .. code-block:: python

        x_val = '[1, 2, 3, 4]'
        x_val = convert_str_float_list_vector(x_val)

    Args:
        x_val (str): String with the values separated by commas.

    Returns:
        np.ndarray: Vector with the values.
    """

    x_val = (
        x_val.replace("[", "")
        .replace("]", "")
        .replace("\n", ",")
        .replace(" ", ",")
        .split(",")
    )
    x_val = np.array([float(x) for x in x_val if x != ""])
    return x_val


def line_intersection(
    line1: np.ndarray, line2: np.ndarray
) -> Tuple[float, float]:
    """find the intersection of two lines.

    example:

    .. code-block:: python

        line1 = np.array([[0, 0], [1, 1]])
        line2 = np.array([[1, 0], [0, 1]])
        x, y = line_intersection(line1, line2)

    Args:
        line1 (np.ndarray): Vector with the coordinates of the first line.
        line2 (np.ndarray): Vector with the coordinates of the second line.

    Raises:
        Exception: Handle the case when the lines do not intersect.

    Returns:
        Tuple[float, float]: x and y coordinates of the intersection.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def kinoshita_curve_abad(
    theta_0: float,
    lambda_value: float,
    j_s: float,
    j_f: float,
    n: int,
    m_points: int = 1000,
    ds: Union[None, float] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate a Kinoshita Curve with the information related
    to the reach generated.
    The Kinoshita curve is based on (Kinoshita, 1961). The equations presented
    in this function are based on the equations presented in
    (Abad and Garcia, 2009).

    Equation:

    .. math::
        \\theta(s) = \\theta_0 \\sin(k s) + \\theta_0^3 (j_s \\cos(3 k s) - j_f \\sin(3 k s))

    example:

    .. code-block:: python

        x, y, data = kinoshita_curve_abad(
            theta_0=110*np.pi/180,
            lambda_value=100,
            j_s=0.020,
            j_f=0,
            n=3,
            m_points=1000,
            ds=None
        )

    References:
        Abad, J. D., & Garcia, M. H. (2009). Experiments in a
        high-amplitude Kinoshita meandering channel: 1. Implications
        of bend orientation on mean and turbulent flow structure:
        KINOSHITA CHANNEL, 1. Water Resources Research, 45(2).
        https://doi.org/10.1029/2008WR007016

        Kinoshita, R. (1961). Investigation of channel
        deformation in Ishikari River. Report of Bureau of
        Resources, 174. Retrieved from
        https://cir.nii.ac.jp/crid/1571417124444824064


    Args:
        theta_0 (float): Maximum angular amplitude in radians.
        lambda_value (float): Arc wavelength.
        j_s (float): Skewness.
        j_f (float): Flatness or "fatness".
        n (int): Number of meander loops.
        m_points (int, optional): Number of points that describe the meander.
            This parameter would be overwritten if ds is provided.
            Defaults to 1000.
        ds (Union[None, float], optional): delta of distance. If this parameter
            is None the function will calculate the distance between points
            using the number of pints (m_points). Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: X and Y coordinates and a
            dictionary with the curvature, theta, streamwise coordinates,
            maximum length, maximum y extent, and sinuosity. The dictionary
            contains the following keys
            - 'curve': curvature,
            - 'theta': values of theta in each iteration,
            - 's': streamwise coordinates,
            - 'lmax': maximum length,
            - 'ymax': maximum y extent,
            - 'sinuosity': sinuosity (sigma = smax/lmax).
    """

    # Direction
    smax = n * lambda_value
    if ds is None:
        s = np.linspace(0, smax, m_points)
    else:
        s = np.arange(0, smax + ds, ds)
    deltas = s[1]

    k = 2 * np.pi / lambda_value

    theta_rad = theta_0 * np.sin(k * s) + theta_0**3 * (
        j_s * np.cos(3 * k * s) - j_f * np.sin(3 * k * s)
    )
    theta_rad = theta_rad[:-1]
    theta = theta_rad * 180 / np.pi

    curve = (
        k * theta_0 * np.cos(k * s)
        - 3 * k * theta_0**3 * (j_s * np.sin(3 * k * s))
        + j_f * np.cos(3 * k * s)
    )

    # Generate coordinates
    deltax = deltas * np.cos(theta_rad)
    deltay = deltas * np.sin(theta_rad)

    x = np.array([0] + list(np.cumsum(deltax)))
    y = np.array([0] + list(np.cumsum(deltay)))

    lmax = x[-1]
    ymax = np.max(np.abs(y))
    sinuosity = smax / lmax

    data = {
        "curve": curve,
        "theta": theta,
        "s": s,
        "lmax": lmax,
        "ymax": ymax,
        "sinuosity": sinuosity,
    }

    return x, y, data


def kinoshita_curve_zolezzi(
    theta_0: float,
    lambda_value: float,
    theta_s: float,
    theta_f: float,
    n: int,
    m_points: int = 1000,
    ds: Union[None, float] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Generate a Kinoshita Curve with the information related to reach
    generated.

    The Kinoshita curve is based on (Kinoshita, 1961). The
    equations presented in this function are based on the
    equations presented in (Zolezzi and Güneralp, 2016).

    Equation:

    .. math::
        \\theta(s) = \\theta_0 \\cos(k s) + \\theta_s \\sin(3 k s) + \\theta_f \\cos(3 k s)

    example:

    .. code-block:: python

        x, y, data = kinoshita_curve_zolezzi(
            theta_0=110*np.pi/180,
            lambda_value=100,
            theta_s=0.344,
            theta_f=0.031,
            n=3,
            m_points=1000,
            ds=None
        )

    References:

    Kinoshita, R. (1961). Investigation of channel
    deformation in Ishikari River. Report of Bureau of
    Resources, 174. Retrieved from
    https://cir.nii.ac.jp/crid/1571417124444824064

    Zolezzi, G., & Güneralp, I. (2016). Continuous wavelet
    characterization of the wavelengths and regularity of
    meandering rivers. Geomorphology, 252, 98–111.
    https://doi.org/10.1016/j.geomorph.2015.07.029

    Args:
        theta_0 (float): Maximum angular amplitude in radians.
        lambda_value (float): Arc wavelength.
        theta_s (float): coefficient for Skewness in radians.
        theta_f (float): coefficient for Fatness in radians.
        n (int): Number of loops.
        m_points (int, optional): Number of points that describe the meander.
            This parameter would be overwritten if ds is provided.
            Defaults to 1000.
        ds (Union[None, float], optional): delta of distance. If this parameter
            is None the function will calculate the distance between points
            using the number of pints (m_points). Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, dict]: X and Y coordinates and a
            dictionary with the curvature, theta, streamwise coordinates,
            maximum length, maximum y extent, and sinuosity. The dictionary
            contains the following keys
            - 'curve': curvature,
            - 'theta': values of theta in each iteration,
            - 's': streamwise coordinates,
            - 'lmax': maximum length,
            - 'ymax': maximum y extent,
            - 'sinuosity': sinuosity (sigma = smax/lmax).
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
        theta_0 * np.cos(k * s)
        + theta_s * np.sin(3 * k * s)
        + theta_f * np.cos(3 * k * s)
    )

    curve = -k * (
        theta_0 * np.sin(k * s)
        - 3 * theta_s * np.cos(3 * k * s)
        + 3 * theta_f * np.sin(3 * k * s)
    )

    theta = theta_rad * 180 / np.pi

    # Generate coordinates
    deltax = deltas * np.cos(theta_rad)
    deltay = deltas * np.sin(theta_rad)

    x = np.array([0] + list(np.cumsum(deltax)))
    y = np.array([0] + list(np.cumsum(deltay)))

    lmax = x[-1]
    ymax = np.max(np.abs(y))
    sinuosity = smax / lmax

    data = {
        "c": curve,
        "theta": theta,
        "s": s,
        "lmax": lmax,
        "ymax": ymax,
        "sinuosity": sinuosity,
    }

    return x, y, data


def rle(in_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run length encoding. Partial credit to R rle function.
    Multi datatype arrays catered for including non Numpy.

    Args:
        in_array (np.ndarray): Array with values

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: return
            z: np.ndarray, run lengths.
            p: np.ndarray, start positions.
            ia: np.ndarray, values.
    """
    ia = np.asarray(in_array)  # force numpy
    n = len(ia)
    if n == 0:
        return None, None, None
    else:
        y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
        i = np.append(np.where(y), n - 1)  # must include last element posi
        z = np.diff(np.append(-1, i))  # run lengths
        p = np.cumsum(np.append(0, z))[:-1]  # positions
        return z, p, ia[i]


def calculate_curvature(
    ss: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    derivatives: Union[dict, None] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate curvature and the direction angle from the coordinates and
    the arc-length of the river transect.

    The equation for curvature and direction angle are based on the
    equations presented in (Güneralp and Rhoads, 2008).

    If the derivatives are not provided, the function will calculate them
    using the np.gradient function.

    Equation:

    .. math::

        C = \\frac{x'y''-y'x''}{[(x')^2+(y')^2]^{3/2}}

    example:

    .. code-block:: python

        ss = np.linspace(0, 100, 100)
        xs = np.sin(ss)
        ys = np.cos(ss)
        r, c, theta = calculate_curvature(ss, xs, ys)

    References:

    Güneralp, İ., & Rhoads, B. L. (2008). Continuous Characterization of the
    Planform Geometry and Curvature of Meandering Rivers. Geographical
    Analysis, 40(1), 1–25. https://doi.org/10.1111/j.0016-7363.2007.00711.x

    Args:
        ss (np.ndarray): streamwise coordinates.
        xs (np.ndarray): x coordinates.
        ys (np.ndarray): y coordinates.
        derivatives (Union[dict, None], optional): derivatives of the
            coordinates. If None the function will calculate them using
            np.gradient function. The dictionary must contain the following
            keys:
            - 'dxds': derivative of x with respect to s.
            - 'dyds': derivative of y with respect to s.
            - 'd2xds2': second derivative of x with respect to s.
            - 'd2yds2': second derivative of y with respect to s
            Defaults to None.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: curvature, radius of
            curvature, and direction angle.
    """
    if derivatives is None:
        dx = np.gradient(xs, ss)
        dy = np.gradient(ys, ss)
        d2x = np.gradient(dx, ss)
        d2y = np.gradient(dy, ss)
    else:
        dx = derivatives["dxds"]
        dy = derivatives["dyds"]
        d2x = derivatives["d2xds2"]
        d2y = derivatives["d2yds2"]
    c = (dx * d2y - dy * d2x) / (dx**2 + dy**2) ** (3 / 2)
    c_r = copy.deepcopy(c)
    c_r[c_r == 0] = np.nan
    r = -1 / c_r

    # --------------------------------
    # Calculate direction angle
    # --------------------------------
    segm_length = np.diff(ss)
    theta = np.zeros_like(ss)
    # Start with known point
    # theta[0] = np.arctan(dy/dx)[0]
    # Direction-angle
    alpha = np.arctan(dy / dx)[0]
    # Conditions
    if dy[0] > 0 and dx[0] < 0:
        # Condition 2
        theta[0] = np.pi + alpha
    elif dy[0] < 0 and dx[0] < 0:
        # Condition 4
        theta[0] = 2 * np.pi - np.pi / 2 - alpha
    else:
        # Condition 1
        theta[0] = copy.deepcopy(alpha)

    theta[1:] = c[1:] * segm_length
    theta = np.cumsum(theta)
    return r, c, theta


def get_inflection_points(
    s: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Obtain the inflection points from the curvature.

    example:

    .. code-block:: python

        s = np.linspace(0, 100, 100)\\
        c = np.sin(s)\\
        s_inf, c_inf = get_inflection_points(s, c)\\

    Args:
        s (np.ndarray): streamwise coordinates.
        c (np.ndarray): Curvature.

    Returns:
        Tuple[np.ndarray, np.ndarray]: streamwise coordinates, curvature,
            left curvature inflection point, and right curvature inflection
            point.
    """

    # Find inflexion points
    # condition_c = (c >= 0)
    condition_c = c > 0

    lengths, positions, type_run = rle(condition_c)

    ind_l = positions[1:] - 1
    ind_r = positions[1:]

    x_l = s[ind_l]
    x_r = s[ind_r]
    y_l = c[ind_l]
    y_r = c[ind_r]

    # Get Inflection points
    s_inf = -(x_r - x_l) / (y_r - y_l) * y_l + x_l
    c_inf = np.zeros_like(s_inf)
    return s_inf, c_inf, ind_l, ind_r


def calculate_direction_angle(
    ss: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    derivatives: Union[dict, None] = None,
) -> np.ndarray:
    """Calculate the direction angle from the coordinates and the arc-length.
    Keep in mind that this calculation would not work if the river direction
    is in the second and third cartesian quadrants from the start of the river.

    To have a better estimate of the direction angle use the function
    :func:`RiverFunctions.calculate_curvature`.

    Equation:

    .. math::

        \\theta = \\theta_0 + \\int_{s=0}^{s=s_n}Cds

    .. code-block:: python

        # Calculate direction angle
        ss = np.linspace(0, 100, 100)
        xs = np.sin(ss)
        ys = np.cos(ss)
        theta = calculate_direction_angle(ss, xs, ys)
        print(theta)

    Args:
        ss (np.ndarray): streamwise coordinates.
        xs (np.ndarray): x coordinates.
        ys (np.ndarray): y coordinates.
        derivatives (Union[dict, None], optional): Dictionary with the
            derivatives of the coordinates with respect to the arc-length.
            Defaults to None.

    Returns:
        np.ndarray: Direction angle.
    """
    if derivatives is None:
        dxds = np.gradient(xs, ss)
        dyds = np.gradient(ys, ss)
    else:
        dxds = derivatives["dxds"]
        dyds = derivatives["dyds"]

    # -------------------------------------------------------------------------
    # These computations have complications when the river is rotated. Making
    #   the angle jump between positive and negative angles once we complete
    #   a loop.

    # Direction-angle
    alpha = np.arctan(dyds / dxds)
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


def calculate_direction_azimuth(
    ss: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    derivatives: Union[dict, None] = None,
) -> np.ndarray:
    """Calculate the direction azimuth from the coordinates and the arc-length.
    Keep in mind that this calculation would not work if the river direction is
    in the second and third cartesian plane quadrants from the start of the
    river.

    To have a better estimate of the direction angle use the function
    :func:`RiverFunctions.calculate_curvature`.

    Equation:

    .. math::

        \\theta = \\tan^{-1}\left(\\frac{y'}{x'}\\right)

    example:

    .. code-block:: python

        ss = np.linspace(0, 100, 100)
        xs = np.sin(ss)
        ys = np.cos(ss)
        theta = calculate_direction_azimuth(ss, xs, ys)

    Args:
        ss (np.ndarray): arc-length
        xs (np.ndarray): x coordinates
        ys (np.ndarray): y coordinates
        derivatives (Union[dict, None], optional): Dictionary with the
            derivatives of the coordinates with respect to the arc-length.
            Defaults to None.

    Returns:
        np.ndarray: Direction azimuth.
    """
    if derivatives is None:
        dxds = np.gradient(xs, ss)
        dyds = np.gradient(ys, ss)
    else:
        dxds = derivatives["dxds"]
        dyds = derivatives["dyds"]

    # -------------------------------------------------------------------------
    # These computations have complications when the river is rotated. Making
    #   the angle jump between positive and negative angles once we complete
    #   a loop.
    # Azimuth
    # For cuadrant 1
    alpha = np.arctan(dyds / dxds)
    theta = np.pi / 2 - alpha

    # For cuadrant 2
    cond = (dyds >= 0) & (dxds < 0)
    theta[cond] = 3 * np.pi / 2 - alpha[cond]

    # For cuadrant 3
    cond = (dyds < 0) & (dxds > 0)
    theta[cond] = np.pi / 2 - alpha[cond]

    # For cuadrant 4
    cond = (dyds < 0) & (dxds < 0)
    theta[cond] = 3 * np.pi / 2 - alpha[cond]
    return theta


def translate(p: np.ndarray, p1: np.ndarray) -> np.ndarray:
    """translate points (p) with respect to p1.

    example:

    .. code-block:: python

        p = np.array([[1, 1], [1, 2], [2, 2]])
        p1 = np.array([1, 1])
        p_trans = translate(p, p1)
        print(p_trans)

    Args:
        p (np.ndarray): points to translate
        p1 (np.ndarray): point to translate with respect to

    Returns:
        np.ndarray: translated points
    """
    return p - p1


def rotate(
    p: np.ndarray,
    p1: Union[np.ndarray, None] = None,
    p2: Union[np.ndarray, None] = None,
    theta: Union[float, None] = None,
) -> Tuple[np.ndarray, float]:
    """rotate points (p) an angle theta or rotate the points such that p1 and
    p2 are aligned with the x-axis. The angle theta is calculated from the
    points p1 and p2. The rotation is done with the following matrix:

    .. math::
        \\begin{bmatrix}
            \\cos(\\theta) & \\sin(\\theta) \\\\
            -\\sin(\\theta) & \\cos(\\theta)
        \\end{bmatrix}

    example:

    .. code-block:: python
    
        # Rotate points based on p1 and p2
        p = np.array([[1, 1], [1, 2], [2, 2]])
        p1 = np.array([1, 1])
        p2 = np.array([2, 2])
        p_rot, theta = rotate(p, p1, p2)
        print(p_rot)
        print(theta)

        # Rotate points based on theta
        p = np.array([[1, 1], [1, 2], [2, 2]])
        p_rot, theta = rotate(p, theta=np.pi/2)
        print(p_rot)

    Args:
        p (np.ndarray): original coordinates as (n_points, n_variables)
        p1 (Union[np.ndarray, None], option): initial coordinates as
            (1, n_variables). Defaults to None.
        p2 (Union[np.ndarray, None], option): ending coordinates as
            (1, n_variables). Defaults to None. 
        theta (Union[float, None], optional): Angle of rotation. If None the
            code will calculate the angle from p1 and p2 and rotate the points
            such that p1 and p2 are aligned with the x-axis.
            If theta is provided, the code will rotate the points with theta.
            Defaults to None.

    Returns:
        Tuple[np.ndarray, float]: rotated points and the angle theta.
    """

    if theta is None:
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


def get_reach_distances(x_coord: np.ndarray) -> np.ndarray:
    """This function calculates the cummulative streamwise distance of the
    river transect using the coordinates.

    example:

    .. code-block:: python

        x_coord = np.array([[1, 1], [1, 2], [2, 2]])
        s = get_reach_distances(x_coord)
        print(s)

    Args:
        x_coord (np.ndarray): [x, y] coordinates in (n_points, 2)

    Returns:
        np.ndarray: Distance from the start point to the end point.
    """
    s_diff = np.diff(x_coord, axis=0)
    s_dist = np.sqrt((s_diff**2).sum(axis=1))
    s_dist = np.hstack((0, s_dist))
    s = np.cumsum(s_dist)
    return s


def fit_splines(
    s: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    method: str = "geometric_mean",
    ds: float = 0,
    k: int = 3,
    smooth: int = 0,
    ext: int = 0,
    return_derivatives: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """This function fits a spline to the coordinates with the minimum distance
    of the river.

    example:

    .. code-block:: python

        s = np.linspace(0, 100, 100)
        x = np.sin(s)
        y = np.cos(s)
        s_poly, x_poly, y_poly = fit_splines(s, x, y)

    Args:
        s (np.ndarray): Streamwise distance
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.
        method (str, optional): Method to use for fitting the spline.
            The method can be "min", "geometric_mean", or "mean". Defaults to
            "geometric_mean".
        ds (float, optional): Distance between points in the spline.
            This parameter overrides the method parameter. Defaults to 0.
        k (int, optional): Degree of the spline. Defaults to 3.
        smooth (int, optional): Smoothness of the spline. Defaults to 0.
        ext (int, optional): Number of points to extrapolate. Defaults to 0.
        return_derivatives (bool, optional): Whether to return derivatives. Defaults to True.

    Raises:
        ValueError: If method is not recognized.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:Tuple
            of the spline points, x coordinates, y coordinates, derivatives
            of the spline, and derivatives of the x coordinates.
    """
    method = method.lower()
    # ------------------
    # Fit the splines
    # ------------------
    if method == "min" or method == "minimum":
        diff_s = np.min(np.diff(s))
    elif method == "geometric_mean":
        diff_s = 10 ** np.mean(np.log10(np.diff(s)))
    elif method == "mean":
        diff_s = np.mean(np.diff(s))
    else:
        raise ValueError(
            f"method '{method} not implemented."
            f"Please use 'min' or 'geometric_mean'"
        )
    if ds > 0:
        diff_s = ds

    s_poly = np.arange(s[0], s[-1] + diff_s / 2, diff_s)
    # ------------------
    # Generate Splines
    # -----------------
    x_spl = UnivariateSpline(s, x, k=k, s=smooth, ext=ext)
    y_spl = UnivariateSpline(s, y, k=k, s=smooth, ext=ext)
    x_poly = x_spl(s_poly)
    y_poly = y_spl(s_poly)
    # s_poly_2 = get_reach_distances(np.vstack((x_poly, y_poly)).T)
    splines = {"x_spl": x_spl, "y_spl": y_spl}

    if return_derivatives:
        if k > 1:
            dxds = x_spl.derivative(n=1)(s_poly)
            dyds = y_spl.derivative(n=1)(s_poly)
            d2xds2 = x_spl.derivative(n=2)(s_poly)
            d2yds2 = y_spl.derivative(n=2)(s_poly)
            derivatives = {
                "dxds": dxds,
                "dyds": dyds,
                "d2xds2": d2xds2,
                "d2yds2": d2yds2,
            }
        else:
            # print('Fitted spline is linear, '
            #       'calculating derivatives with np.gradient')
            dxds = np.gradient(x_poly, s_poly)
            dyds = np.gradient(y_poly, s_poly)
            d2xds2 = np.gradient(dxds, s_poly)
            d2yds2 = np.gradient(dyds, s_poly)
            derivatives = {
                "dxds": dxds,
                "dyds": dyds,
                "d2xds2": d2xds2,
                "d2yds2": d2yds2,
            }
        return s_poly, x_poly, y_poly, derivatives, splines
    else:
        return s_poly, x_poly, y_poly


def fit_splines_complete(
    data: dict,
    method: str = "geometric_mean",
    ds: float = 0,
    k: int = 3,
    smooth: float = 0,
    ext: int = 0,
) -> dict:
    """function to fit splines to the data of the River class.

    example:

    .. code-block:: python

        data = {
            "comid": comid,
            "so": so,
            "s": s,
            "x": x,
            "y": y,
            "z": z,
            "da_sqkm": da,
            "w_m": w,
        }
        splines = fit_splines_complete(data)

    Args:
        data (dict): dictionary containing the data of the river
        method (str, optional): Method to use for fitting the spline.
            The method can be "min", "geometric_mean", or "mean". Defaults to
            "geometric_mean".
        ds (float, optional): Distance between points in the spline.
            This parameter overrides the method parameter. Defaults to 0.
        k (int, optional): Degree of the spline. Defaults to 3.
        smooth (int, optional): Smoothness of the spline. Defaults to 0.
        ext (int, optional): Number of points to extrapolate. Defaults to 0.

    Raises:
        ValueError: if method is not implemented

    Returns:
        dict: dictionary containing the splines
    """
    # Extract data
    comid = np.array(data["comid"])
    so = data["so"]
    s = data["s"]
    x = data["x"]
    y = data["y"]
    z = data["z"]
    da = data["da_sqkm"]
    w = data["w_m"]
    # Set smooth relative to the length of the data
    smooth = smooth * len(s)
    # ----------------------------------------
    # Calculate geometric meand of the width
    # ----------------------------------------
    w_gm = 10 ** np.mean(np.log10(w))
    w_value = np.nanmin(w)
    if method == "min_width" and not (np.isnan(w_value)):
        method = "geometric_mean"
        ds = w_value
    elif method == "geometric_mean_width" and not (np.isnan(w_gm)):
        method = "geometric_mean"
        ds = w_gm
    elif method in ["min_width", "geometric_mean_width"] and np.isnan(w_value):
        raise ValueError("The width value is NaN")
    # -------------------
    # Get coordinate poly
    # -------------------
    s_poly, x_poly, y_poly, derivatives, splines = fit_splines(
        s,
        x,
        y,
        method=method,
        ds=ds,
        k=k,
        smooth=smooth,
        ext=ext,
        return_derivatives=True,
    )
    # ----------------------------------------
    # Generate Splines on the rest of the data
    # ----------------------------------------
    # x_spl = UnivariateSpline(s, x, k=k, s=smooth, ext=ext)
    # y_spl = UnivariateSpline(s, y, k=k, s=smooth, ext=ext)

    z_spl = UnivariateSpline(s, z, k=1, s=0, ext=0)
    f_comid = interpolate.interp1d(
        s,
        comid,
        fill_value=(comid[0], comid[-1]),
        kind="previous",
        bounds_error=False,
    )
    f_so = interpolate.interp1d(
        s, so, fill_value=(so[0], so[-1]), kind="previous", bounds_error=False
    )
    f_da = interpolate.interp1d(s, da, fill_value="extrapolate")
    f_w = interpolate.interp1d(s, w, fill_value="extrapolate")
    splines.update(
        {
            "z_spl": z_spl,
            "f_comid": f_comid,
            "f_so": f_so,
            "f_da": f_da,
            "f_w": f_w,
        }
    )
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
        "s_poly": s_poly,
        "x_poly": x_poly,
        "y_poly": y_poly,
        "z_poly": z_poly,
        "comid_poly": comid_poly,
        "so_poly": so_poly,
        "da_sqkm_poly": da_poly,
        "w_m_poly": w_poly,
        "derivatives": derivatives,
        "splines": splines,
    }

    return data_fitted


def smooth_data(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    poly_order: int = 2,
    savgol_window: int = 2,
    gaussian_window: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """smooth the data using savgol and gaussian filters.

    example:

    .. code-block:: python

        x = np.linspace(0, 100, 100)
        y = np.sin(x)
        s = np.linspace(0, 100, 100)
        s_smooth, x_smooth, y_smooth = smooth_data(x, y, s)

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
        s (np.ndarray): streamwise coordinates
        poly_order (int, optional): order of the polynomial. Defaults to 2.
        savgol_window (int, optional): window size for savgol filter. Defaults to 2.
        gaussian_window (int, optional): window size for gaussian filter. Defaults to 1.

    Raises:
        ValueError: if savgol_window is not odd

    Returns:
        s_smooth (np.ndarray): smoothed streamwise coordinates
        x_smooth (np.ndarray): smoothed x coordinates
        y_smooth (np.ndarray): smoothed y coordinates
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
    s_smooth, x_smooth, y_smooth = fit_splines(
        s_smooth, x_smooth, y_smooth, ds=ds_new, return_derivatives=False
    )

    if len(x_smooth) != len(x):
        raise ValueError("The length of the smoothed data is different")

    return s_smooth, x_smooth, y_smooth


def calculate_lambda(x: np.ndarray, y: np.ndarray) -> float:
    """calculate the lenth of the meander.

    Equation:

    .. math::

        \\lambda = \\sum_{i=j}^k\\sqrt{(x_{i+1}-x_{i})^2+(y_{i+1}-y_{i})^2}

    example:

    .. code-block:: python

        x = np.linspace(0, 100, 100)
        y = np.sin(x)
        l = calculate_lambda(x, y)

    Args:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.

    Returns:
        np.ndarray: length of the meander.
    """

    coords = np.vstack((x, y)).T
    s_calc = get_reach_distances(coords)
    return s_calc[-1]


def calculate_l(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate valley length.

    Equation:

    .. math::

        l = \\sqrt{(x_{end}-x_{start})^2+(y_{end}-y_{start})^2}

    example:

    .. code-block:: python

        x = np.linspace(0, 100, 100)
        y = np.sin(x)
        l = calculate_l(x, y)

    Args:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.

    Returns:
        float: valley length.
    """
    l = np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2)
    return l


def calculate_sinuosity(l: float, lambda_value: float) -> float:
    """Calculate the sinuosity.

    Equation:

    .. math::

        sinuosity = \\frac{\\lambda}{l}

    example:

    .. code-block:: python

        l = calculate_l(x, y)
        lambda_value = calculate_lambda(x, y)
        sinuosity = calculate_sinuosity(l, lambda_value)

    Args:
        l (float): valley length.
        lambda_value (float): meander length.

    Returns:
        float: sinuosity.
    """
    # Check valley distance
    if l == 0:
        sinuosity = np.nan
    else:
        sinuosity = lambda_value / l
    return sinuosity


def calculate_radius_of_curvature(
    x: np.ndarray, y: np.ndarray, wavelength: float
) -> Tuple[float, float, float]:
    """Calculate the radius of curvature of the meander by fitting a circle
    to the half-meander section and using the wavelength as the arc length.

    Equation:

    .. math::

        \\frac{1}{R} = \\frac{\\lambda}{2 \\pi w}

    example:

    .. code-block:: python

        x = np.linspace(0, 100, 100)
        y = np.sin(x)
        wavelength = 100
        x_c, y_c, radius = calculate_radius_of_curvature(x, y, wavelength)

    Args:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.
        wavelength (float): Wavelength of the meander.

    Returns:
        Tuple[float, float, float]: x and y coordinates of the center and radius
            of curvature.
    """
    coordinates = np.vstack((x, y)).T
    s_val = get_reach_distances(coordinates)
    s_mid = s_val[-1] / 2
    arg_mid = np.argmin(np.abs(s_val - s_mid))
    x_mid = x[arg_mid]
    y_mid = y[arg_mid]
    # --------------------------
    # Fit Circle
    # --------------------------
    x_cen, y_cen, r, sigma = taubinSVD(coordinates)

    # Calculate Omega
    w = wavelength / (2 * np.pi)
    rvec = np.array([x_cen - x_mid, y_cen - y_mid]) / r
    x_c = x_mid + rvec[0] * w
    y_c = y_mid + rvec[1] * w
    radius = np.sqrt((x_c - x_mid) ** 2 + (y_c - y_mid) ** 2)
    return x_c, y_c, radius


def calculate_asymetry(
    x: np.ndarray, y: np.ndarray, c: np.ndarray
) -> Tuple[float, float, float]:
    """Calculate the asymmetry of the meander using Eq. 24 in
    Howard and Hemberger (1991).

    If the value is lower than zero the meander has an assymetry to the
    left, and if the value is higher than zero the meander has an
    assymetry to the right. For most NHDPlus information cases
    left is upstream and right is downstream.

    Equation:

    .. math::

        a = \\frac{\lambda_u - \lambda_d}{\lambda}

    example:

    .. code-block:: python

        x = np.linspace(0, 100, 100)
        y = np.sin(x)
        c = calculate_curvature(x, y)
        a = calculate_asymetry(x, y, c)

    References:

    Howard, A. D., & Hemberger, A. T. (1991). Multivariate characterization
    of meandering. Geomorphology, 4(3–4), 161–186.
    https://doi.org/10.1016/0169-555X(91)90002-R

    Args:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.
        c (np.ndarray): curvature of the meander.

    Returns:
        Tuple[float, float, float]: asymmetry of the meander, half-length of
            the meander, length of the upper part of the meander, length of
            the lower part of the meander.
    """
    # Detect maximum point of curvature
    argmax_c = np.argmax(np.abs(c))
    # Calculate distances
    lambda_h = calculate_lambda(x, y)
    lambda_u = calculate_lambda(x[: argmax_c + 1], y[: argmax_c + 1])
    lambda_d = calculate_lambda(x[argmax_c:], y[argmax_c:])

    # Calculate assymetry
    a_h = (lambda_u - lambda_d) / lambda_h
    return a_h, lambda_h, lambda_u, lambda_d


def extend_node_bound(node: Node, c: np.ndarray) -> Node:
    """Extend the bounds of a node in the meanders.

    Args:
        node (Node): Node of the meanders.
        c (np.ndarray): Curvature of the transect.

    Returns:
        Node: Node with idx_planimetry_extended_start and
              idx_planimetry_extended_end.
    """
    #  curvature of the adjacent meanders.
    # ------------------------------
    # Extract Information
    # ------------------------------
    idx_start = node.idx_planimetry_start
    idx_end = node.idx_planimetry_end
    c_meander = c[idx_start : idx_end + 1]
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
    # dif_c = np.abs(max_peak - min_peak)

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
        val_range_left = np.arange(idx_left, idx_start + 1, 1).astype(int)
        c_left = mult * c[val_range_left]
        peak_left, _ = find_peaks(c_left)
        # Selected Value with highest curvature
        if len(peak_left) > 0:
            # Find Peaks
            c_at_peaks_left = c_left[peak_left]
            max_c_left = np.max(c_at_peaks_left)
            closer_c_left = c_at_peaks_left[-1]
            # dif_c_left = np.abs(max_c_left - closer_c_left)
            idx_peak_left = val_range_left[c_left == max_c_left][0]
            # # Compare values to pick the best curvature peak
            # if dif_c_left >= 0.2*dif_c:
            #     idx_peak_left = val_range_left[c_left == max_c_left][0]
            # else:
            #     idx_peak_left = val_range_left[peak_left[-1]]
        else:
            idx_peak_left = copy.deepcopy(idx_start)
            idx_dif_left += idx_dif // 2
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
        c_right = mult * c[val_range_right]
        peak_right, _ = find_peaks(c_right)
        # Selected Value with highest curvature
        if len(peak_right) > 0:
            # Find Peaks
            c_at_peaks_right = c_right[peak_right]
            max_c_right = np.max(c_at_peaks_right)
            # closer_c_left = c_at_peaks_right[0]
            # dif_c_right = np.abs(max_c_right - closer_c_left)
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
            # idx_dif_right += idx_dif // 2
        i += 1
        if i > 10:
            break

    # Update node information
    node.idx_planimetry_extended_start = idx_peak_left
    node.idx_planimetry_extended_end = idx_peak_right

    return node


def calculate_coordinates_from_curvature(
    s_curvature: np.ndarray,
    c: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the coordinates from the curvature of the river. The
    coordinates are calculated using a numerical integration method. Using
    the angle between the initial direction.

    Equation:

    .. math::
        x = x_0 + \\int_0^{s_n} C ds

    .. math::

        y = y_0 + \\int_0^{s_n} C ds

    example:

    .. code-block:: python

        s_curvature = np.linspace(0, 100, 100)
        c = np.sin(s_curvature)
        x = np.cos(s_curvature)
        y = np.sin(s_curvature)
        x_r, y_r = calculate_coordinates_from_curvature(s_curvature, c, x, y)

    Args:
        s_curvature (np.ndarray): streamwise coordinates.
        c (np.ndarray): curvature.
        x (np.ndarray): x initial two coordinates.
        y (np.ndarray): y initial two coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: x and y reconstructed coordinates.
    """
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
            np.array(
                [
                    [np.cos(delta_theta), -np.sin(delta_theta)],
                    [np.sin(delta_theta), np.cos(delta_theta)],
                ]
            ),
            current_direction,
        )
        # Update x and y coordinates using direction vector and arc length
        delta_x = arc_length * current_direction[0]
        delta_y = arc_length * current_direction[1]
        current_pos[0] += delta_x
        current_pos[1] += delta_y
        x_r.append(current_pos[0])
        y_r.append(current_pos[1])

    return np.array(x_r), np.array(y_r)


def calculate_channel_width(da: np.ndarray) -> np.ndarray:
    """Calculate the channel width from the drainage area.

    This function uses equation (15) presented in Wilkerson et al. (2014).

    example:

    .. code-block:: python

        da = 100
        w = calculate_channel_width(da)

    References:

    Wilkerson, G. V., Kandel, D. R., Perg, L. A., Dietrich, W. E., Wilcock,
    P. R., & Whiles, M. R. (2014). Continental-scale relationship between
    bankfull width and drainage area for single-thread alluvial channels.
    Water Resources Research, 50. https://doi.org/10.1002/2013WR013916

    Args:
        da (np.ndarray): Drainage area in km^2.

    Returns:
        np.ndarray: Channel width in m.
    """
    if isinstance(da, int) or isinstance(da, float):
        da = np.array([da])

    cond_1 = np.log(da) < 1.6
    cond_2 = (np.log(da) >= 1.6) & (np.log(da) < 5.820)
    cond_3 = np.log(da) >= 5.820

    w = np.zeros_like(da)

    w[cond_1] = 2.18 * da[cond_1] ** 0.191
    w[cond_2] = 1.41 * da[cond_2] ** 0.462
    w[cond_3] = 7.18 * da[cond_3] ** 0.183

    # Threshold
    w[w < 1] = 1

    return w


def calculate_spectrum_cuts(
    s: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate the spectrum cuts of the curvature.

    example:

    .. code-block:: python

        s = np.linspace(0, 100, 100)
        c = np.sin(s)
        peaks_min, min_s = calculate_spectrum_cuts(s, c)

    Args:
        s (np.ndarray): Streamwise distance vector.
        c (np.ndarray): Curvature.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple of the indices of the minima of the
            curvature and the streamwise distance of the minima of the curvature.
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
    values = WTFunc.calculate_cwt(c, ds, mother="MORLET")
    wave = values[0]

    wave = np.abs(wave**2)
    wave_sum = np.sum(wave, axis=0)
    peaks_min, _ = find_peaks(-wave_sum, prominence=np.std(wave_sum))
    min_s = s[peaks_min]
    return peaks_min, min_s


def calculate_amplitude(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate amplitude of the meanders. This funcion rotates the meanders
    and calculates the distance between the maximum and minimum y values.

    Additionally, it calculates the dimensionless coordinate of the meanders
    $x^*$ that depends on the angle between the starting point of the meander
    and each point within the meander.

    Equation:

    .. math:: A = \\max(y) - \\min(y)

    .. math:: x^* = \\frac{\\beta}{\\pi} - 0.5

    example:

    .. code-block:: python

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        amplitude = calculate_amplitude(x, y)

    Args:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.

    Returns:
        np.ndarray: Amplitude of the meanders.
    """
    # Calculate the distance
    coords = np.vstack((x, y)).T
    index_initial = 0
    index_final = len(coords) - 1
    rotated_points, theta = translate_rotate(
        coords, index_initial=index_initial, index_final=index_final
    )
    # --------------------------
    # Calculate amplitude
    # --------------------------
    y_rot = rotated_points[:, 1]
    if y_rot[len(y_rot) // 2] < 0:
        y_rot = -y_rot
        rotated_points[:, 1] = y_rot
    amplitude = np.max(y_rot) - np.min(y_rot)

    # ------------------------------------------------------------
    # Calculate the dimensionless coordinate
    # ------------------------------------------------------------
    x_apex = rotated_points[np.argmax(y_rot), 0]
    # translate the points to the origin
    rotated_points[:, 0] -= x_apex
    # Locate center point
    y_center = 0.0
    x_center = 0.0

    # Calculate the angle between the start and each point
    angle_st_all = np.arctan2(rotated_points[:, 1], rotated_points[:, 0])
    # Correct angles
    if angle_st_all[0] < 0:
        angle_st_all[0] += 2 * np.pi  # Correct cuadrant of the first point
    angle_st_all = np.pi - angle_st_all  # Correct sign of the angle

    # Calculate the dimensionless coordinate
    x_star = (angle_st_all / np.pi) - 0.5

    # find center point in the original coordinates
    center_point = np.array([x_center, y_center])
    # translate the center point to the origin
    center_point[0] += x_apex
    # Translate to the original coordinates
    apex_idx = np.argmax(y_rot)
    center_point += coords[index_initial]
    coords_apex = coords[apex_idx]

    # ------------------------------------------------------------
    # TEST
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(x, y, "k-")
    # plt.scatter(x, y, c=x_star, cmap="RdBu_r")
    # plt.scatter(coords_apex[0], coords_apex[1], color="r")
    # plt.scatter(center_point[0], center_point[1], color="b")
    # plt.colorbar()
    # plt.show()
    # ------------------------------------------------------------
    return amplitude, x_star, center_point, coords_apex


def calculate_funneling_factor(
    x: np.ndarray,
    y: np.ndarray,
    s: np.ndarray,
    idx_st: int,
    idx_end: int,
) -> dict:
    """Calculate the funneling factor of the meander. The funneling factor is
    calculated by dividing the distance around the lobe by the distance between
    the inflection points.

    Equation:

    .. math:: FF = \\frac{L_l}{L_n}

    example:

    .. code-block:: python

        x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        s = np.linspace(0, 100, 100)
        idx_st = 0
        idx_end = -1
        results = calculate_funneling_factor(x, y, s, idx_st, idx_end)
        print(results)

    Args:
        x (np.ndarray): x coordinates.
        y (np.ndarray): y coordinates.
        s (np.ndarray): s coordinates.
        idx_st (int): index of the start point.
        idx_end (int): index of the end point.

    Raises:
        ValueError: Raises if meander wraps on itself.
        ValueError: Raises if the funneling factor is too small

    Returns:
        dict: Dictionary with the following keys:
            - FF: float, Funneling factor of the curvature.
            - L_l: float, Length of the lobe.
            - L_n: float, Length of the neck.
            - s_l: float, s-coordinate of the start point of the lobe.
            - s_n: float, s-coordinate of the end point of the neck.
    """
    gen_plots = False
    if gen_plots:
        import matplotlib.pyplot as plt
    # --------------------------------------------------
    # Rotate Values according to the inflection points
    # --------------------------------------------------
    coords = np.vstack((x, y)).T
    rotated_points, _ = translate_rotate(
        coords, index_initial=idx_st, index_final=idx_end
    )
    x_values = rotated_points[:, 0]
    y_values = rotated_points[:, 1]

    # find middle point between the inflection points
    idx_middle = (idx_st + idx_end) // 2
    if y_values[idx_middle] < 0:
        y_values = -y_values
    if x_values[idx_end] < 0:
        x_values = -x_values

    # --------------------------------------------------
    # Fit Spline
    # --------------------------------------------------
    tck, u = splprep([x_values, y_values], u=s, s=0, k=1)
    s = np.linspace(s[0], s[-1], int(np.ceil(len(s) * 1.5)))
    new_points = splev(s, tck)
    x_values = new_points[0]
    y_values = new_points[1]

    # --------------------------
    # Calculate neck distance
    # --------------------------
    y_turning = 0
    dist_l_n_all = []
    lim_y = np.linspace(y_turning, np.min(y_values), 200)
    dist_l_n = 1e9
    l_n = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    s_n = np.array([np.nan, np.nan])
    for i_y_val, y_val in enumerate(lim_y[:-5]):
        i_vals = y_values >= y_val
        x_vals = x_values[i_vals]
        y_vals = y_values[i_vals]
        x_val_1 = x_vals[0]
        x_val_2 = x_vals[-1]
        y_val_1 = y_vals[0]
        y_val_2 = y_vals[-1]
        if gen_plots:
            if i_y_val % 50 == 0:
                plt.figure()
                plt.plot(x_values, y_values)
                # plt.plot(x_vals, y_vals)
                plt.scatter([x_val_1], [y_val_1], color="b")
                plt.scatter([x_val_2], [y_val_2], color="r")
                plt.grid()
                plt.plot([x_val_1, x_val_2], [y_val_1, y_val_2], color="r")
                plt.gca().set_aspect("equal")
                plt.show()
                print((x_val_2 - x_val_1))
                print(np.abs(y_val_2 - y_val_1))

        # if (x_val_2 - x_val_1) < 0:
        #     dist_l_n = np.nan
        #     l_n = np.array([[np.nan, np.nan], [np.nan, np.nan]])
        #     s_n = np.array([np.nan, np.nan])
        #     break
        # elif np.abs(y_val_2 - y_val_1) > 1:
        #     break
        dist_l_n_all.append(x_val_2 - x_val_1)
        if dist_l_n > dist_l_n_all[-1]:
            dist_l_n = dist_l_n_all[-1]
            s_val_12 = s[i_vals][0]
            s_val_22 = s[i_vals][-1]
            x_val_12 = x_values[i_vals][0]
            x_val_22 = x_values[i_vals][-1]
            y_val_12 = y_values[i_vals][0]
            y_val_22 = y_values[i_vals][-1]
            l_n = np.array([[x_val_12, y_val_12], [x_val_22, y_val_22]])
            s_n = np.array([s_val_12, s_val_22])
        if x_val_1 == x_values[0] or x_val_2 == x_values[-1]:
            break

    if np.isnan(l_n[0, 0]):
        dist_l_n = np.nan

    l_l = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    s_l = np.array([np.nan, np.nan])
    if np.isnan(dist_l_n):
        raise ValueError("Meander wraps on itself. Check geometry")
    else:
        # ----------------------------------------
        # Calculate distance inside the meander
        # ----------------------------------------
        dist_l_l_all = []
        lim_y = np.linspace(y_turning, np.max(y_values), 200)
        dist_l_l = 0
        for i_y_val, y_val in enumerate(lim_y):
            x_val_1 = x_values[y_values >= y_val][0]
            x_val_2 = x_values[y_values >= y_val][-1]
            y_val_1 = y_values[y_values >= y_val][0]
            y_val_2 = y_values[y_values >= y_val][-1]
            dist_l_l_all.append(x_val_2 - x_val_1)
            if gen_plots:
                if i_y_val % 30 == 0:
                    plt.figure()
                    plt.plot(x_values, y_values)
                    # plt.plot(x_vals, y_vals)
                    plt.scatter([x_val_1], [y_val_1], color="b")
                    plt.scatter([x_val_2], [y_val_2], color="r")
                    plt.plot([x_val_1, x_val_2], [y_val_1, y_val_2], color="r")
                    plt.gca().set_aspect("equal")
                    plt.show()
            # elif np.abs(y_val_2 - y_val_1) > 1e-3:
            #     break
            if dist_l_l_all[-1] > dist_l_l:
                dist_l_l = dist_l_l_all[-1]
                s_val_1 = s[y_values >= y_val][0]
                s_val_2 = s[y_values >= y_val][-1]
                y_val_1 = y_values[y_values >= y_val][0]
                y_val_2 = y_values[y_values >= y_val][-1]
                l_l = np.array([[x_val_1, y_val_1], [x_val_2, y_val_2]])
                s_l = np.array([s_val_1, s_val_2])

    f_c = dist_l_l / dist_l_n
    if f_c < 1e-5:
        raise ValueError("Funneling factor is too small. Check geometry")
    results = {
        "L_l": dist_l_l,
        "L_n": dist_l_n,
        "L_l_all": l_l,
        "L_n_all": l_n,
        "FF": f_c,
        "s_n": s_n,
        "s_l": s_l,
    }
    return results
