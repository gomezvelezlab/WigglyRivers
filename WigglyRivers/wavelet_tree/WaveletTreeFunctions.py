# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 2023-03-31
# _____________________________________________________________________________
# _____________________________________________________________________________
"""

These functions are based on Vermeulen et al. (2016) Meander tree generation

   Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
   (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
   doi:10.1002/2016GL068238.
"""

# ------------------------
# Importing Modules
# ------------------------
import time
import copy
import numpy as np
from . import waveletFunctions as cwt_func
import matplotlib.pyplot as plt
from matplotlib.path import Path
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from circle_fit import taubinSVD
from ..rivers import RiverFunctions as RF
from ..utilities import utilities as utl
from ..utilities import general_functions as gf
from anytree import Node


# ------------------------
# Functions
# ------------------------
def calculate_cwt(
    curvature, ds, pad=1, dj=5e-2, s0=-1, j1=-1, mother="DOG", m=2
):
    """
    Description:
    ------------
        This function uses package created by Predybaylo (2014), modified by
        von Papen (2018), and is based on the MATLAB package created by Torrence
        and Compo (1998).

        References:

        Torrence, C., & Compo, G. P. (1998). A Practical Guide to Wavelet
        Analysis. Bulletin of the American Meteorological Society, 79(1), 61–78.
        https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2
    ____________________________________________________________________________

    Args:
    ------------
    :param curvature: np.ndarray,
        Curvature of the river.
    :param ds: float,
        Spatial resolution of the curvature.
    :param pad: int,
        pad the time series with zeros, 1 or 0.
    :param dj: float,
        spacing between discrete scales, 1/4 or 1/8.
    :param s0: float,
        smallest scale of the wavelet, 2*dt or 1*dt.
    :param j1: float,
        number of scales minus one, 7/4 or 3/2.
    :param mother: str,
        mother wavelet function, can be 'DOG', 'MORLET', or 'PAUL'.
    :param m: int,
        order of the derivative of the Gaussian, 2 or 4.
    :return:
        wave: real values of the wavelet transform.
        period: period of the wavelet.
        scales: scales of the wavelet.
        coi: cone of influence.
    """
    wave, period, scales, coi, parameters = cwt_func.wavelet(
        curvature, ds, pad, dj, s0, j1, mother, m
    )
    power = np.abs(wave) ** 2

    # Calculate global wavelet spectrum
    gws, peaks = cwt_func.calculate_global_wavelet_spectrum(wave)
    peak_periods = period[peaks]

    # Find SAWP (Spectral-Average Wave Period) using Zolezzi and Guneralp (2016)
    dj = parameters["dj"]
    c_delta = parameters["C_delta"]
    sawp = cwt_func.calculate_scale_averaged_wavelet_power(
        wave, scales, ds, dj, c_delta
    )
    return wave, period, scales, power, gws, peak_periods, sawp, coi, parameters


def find_wave_significance(
    curvature,
    ds,
    scales,
    sigtest=0,
    lag1=0,
    siglvl=0.95,
    dof=None,
    mother="DOG",
    param=None,
    gws=None,
):
    """
    Description:
    ----------------
        Calculate wave significiance.

        This function uses package created by Predybaylo (2014), modified by
        von Papen (2018), and is based on the MATLAB package created by Torrence
        and Compo (1998).

        References:
        
        Torrence, C., & Compo, G. P. (1998). A Practical Guide to Wavelet
        Analysis. Bulletin of the American Meteorological Society, 79(1), 61–78.
        https://doi.org/10.1175/1520-0477(1998)079<0061:APGTWA>2.0.CO;2
    ____________________________________________________________________________

    Args:
    ------------
    :param curvature: np.ndarray,
        Curvature of the river.
    :param ds: float,
        Spatial resolution of the curvature.
    :param scales: np.ndarray,
        Scales of the wavelet.
    :param sigtest: int,
        perform significance test, 0, 1, or 2. Default is 0.
        If 0 (default), then just do a regulat chi-square test, i.e., Eqn (18) from Torrence and Compo (1998).
        If 1, then do a "time-average" test, i.e., Eqn (23).  In this case, DOF should be set to np.nan, the number of local wavelet spectra that were averaged together.  For the Global Wavelet Spectrum, this would be NA=N, where N is the number of points in the time series.
        If 2, then do a "scale-average" test, i.e., Eqn (25)-(28).  In this case, DOF should be set to a two-element vector [S1,S2], which gives the scale range that were averaged together.  For example, if the average between scales 2 and 8 was taken, then DOF=[2,8].
    :param lag1: int,
        lag-1 autocorrelation, used for signif levels. Default is 0.
    :param siglvl: float,
        significance level to use. Default is 0.95.
    :param dof: int,
        degrees of freedom for significance test.
        If sigtest=0, then (automatically) set to 2 (or 1 for mother='DOG').
        If sigtest=1, then set to DOF=np.nan, the number if times averaged.
        If sigtest=2, then set to DOF=[S1,S2], the range of scales averaged.
    :param mother: str,
        mother wavelet function, can be 'DOG', 'MORLET', or 'PAUL'.
    :param param: float,
        parameter for the mother wavelet.
    :param gws: np.ndarray,
        global wavelet spectrum.
    :return:
        - signif: significance levels as a function of scale.
        - sig95: 95% significance level as a function of scale.
    """
    n = len(curvature)
    signif = cwt_func.wave_signif(
        curvature,
        ds,
        scales,
        sigtest=sigtest,
        lag1=lag1,
        siglvl=siglvl,
        dof=dof,
        mother=mother,
        param=param,
        gws=gws,
    )
    # Expand signif --> (J+1)x(N) array
    sig95 = signif[:, np.newaxis].dot(np.ones(n)[np.newaxis, :])
    return signif, sig95


def find_zc_lines(cwt_matrix):
    """
    Description:
    ------------

        Find the zero-crossing lines pairs and the location of the singular
        points in the spectrum.

        Based on Vermeulen et al. (2016) Meander tree generation

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    ------------
    :param cwt_matrix: np.ndarray,
        Wavelet transform of the curvature.
    :return:
        conn: Vector in which each index corresponds to a tree node.
            Roots nodes have a value of -1, while the other nodes have a
            value pointing to the row in conn of their parent node.
        regions: Matrix that contains the same number of indices as conn.
            for each index in conn the corresponding row in regions
            give the scale boundaries of the node and the spatial
            boundaries derived given the scale boundaries derived
            from where the zero crossing lines leave the scale space plane.
            The first column is the smallest bounding period, the second column
            is the largest bounding period, the third column is the spatial
            coordinate where the region starts and the last column is the
            spatial coordinate where the region ends.
        poly: returns a vector of cells with the same size as conn containing
            the coordinates of a polygon bounding the region of the
            corresponding node in conn. Unlike regions these coordinates follow
            the zero crossings lines bounding the region. The coordinates are
            given as [row, col].
        zc_lines: return all zero corssing lines detected. Each element in
            zc_lines contains a Nx2 matrix containing the coordinates of the
            zero crossing line given as [row, col].
        zc_sign: returns the sign of each zero crossing line in zc_lines.
    """
    time_1 = time.time()
    # ============================
    # Flip Matrix
    # ============================
    wave = np.flipud(cwt_matrix)
    # ===========================================
    # Find zero crossing location in spectrum
    # ===========================================
    zcr = np.diff((wave > 0).astype(int), n=1, axis=1)
    zcr = np.hstack((zcr, zcr[:, 0].reshape((-1, 1)) * 0))

    # Create Boundaries in the spectrum
    for row in range(zcr.shape[0]):
        row_vals = zcr[row, :]
        # Left Boundary
        if zcr[row, 0] == 0:
            try:
                zcr[row, 0] = -row_vals[row_vals != 0][0]
            except IndexError:
                zcr[row, 0] = 1
        # Right Boundary
        if zcr[row, -1] == 0:
            zcr[row, -1] = -row_vals[row_vals != 0][-1]

    # ============================
    # Find singular points
    # ============================
    # new zero-crossing lines emerge
    # which mark the edges in scale-direction of the 2d segment
    n_zeros = np.sum(np.abs(zcr), axis=1)
    # Find row (period) where singular points occur (where number of zero
    # crossings change). Note that first row is considered as singular one
    diff_m = np.hstack(([n_zeros[0] - 1], np.diff(n_zeros)))
    singular_points_row = np.where(diff_m > 0)[0]

    # Total number of nodes for variable initialization
    diff = np.diff(zcr[:, [0, -1]], axis=0)
    diff_m = np.vstack(([0, 0], diff)) / 2
    n_new_zc_lines_lateral = np.sum(np.abs(diff_m), axis=1)
    n_new_zc_lines = np.hstack(([0], np.diff(n_zeros)))
    n_new_zc_lines_internal = n_new_zc_lines - n_new_zc_lines_lateral
    n_new_internal_nodes = n_new_zc_lines_internal + n_new_zc_lines_internal / 2

    n_new_lateral_nodes = copy.deepcopy(n_new_zc_lines_lateral)
    n_nodes_intial = n_zeros[0] - 1
    n_nodes_per_period = n_new_internal_nodes + n_new_lateral_nodes
    n_nodes_per_period[0] = n_nodes_intial
    init_n_nodes = np.sum(n_nodes_per_period).astype(int)
    n_zc_lines = n_zeros[0] + np.sum(n_new_zc_lines)

    # =========================================================
    # Make zero-crossing lines from zero-crossing locations
    # =========================================================
    loc_x = np.zeros(wave.shape) * np.nan

    # Initialize Variables
    zc_lines = [0 for _ in range(n_zc_lines * 6)]
    zc_sign = np.zeros(n_zc_lines * 6) * np.nan
    max_row = np.zeros(n_zc_lines * 6) * np.nan
    max_col = np.zeros(n_zc_lines * 6) * np.nan

    # Initialize counters
    last_zc = -1

    # Initialize loop through every period
    for row, s_p in enumerate(singular_points_row):
        zercr_col = np.where(zcr[s_p, :] != 0)[0]
        z_new_col = np.where(np.isnan(loc_x[s_p, zercr_col]))[0]
        col_fin = z_new_col * np.nan
        for col, z_c in enumerate(z_new_col):
            # Get line through
            lrw, lcol = get_zcline(zcr, s_p, zercr_col[z_c])
            if len(lrw) < 2:
                continue
            last_zc += 1
            # Flip line, because initial xwt was flipped
            zc_lines[last_zc] = np.vstack((zcr.shape[0] - lrw - 1, lcol)).T
            # zc_lines[last_zc] = np.vstack((lrw[::-1], lcol)).T
            # Capture sign of the line
            zc_sign[last_zc] = zcr[s_p, zercr_col[z_c]]
            # Column where the line ended
            col_fin[col] = lcol[-1]
            # Save  data in the matrix with the ending column
            loc_x[lrw, lcol] = lcol[-1]
            # Locate singular point location
            max_col[last_zc] = lcol[np.min(lrw) == lrw]
            max_row[last_zc] = zcr.shape[0] - lrw[0] - 1

    idx_nan_st = np.where(np.isnan(zc_sign))[0][0]
    zc_lines = zc_lines[:idx_nan_st]
    zc_sign = zc_sign[:idx_nan_st]
    max_row = max_row[:idx_nan_st]
    max_col = max_col[:idx_nan_st]

    # Flip zc_lines to pick information from lower scales to larger scales
    zc_lines = zc_lines[::-1]
    zc_sign = zc_sign[::-1]
    max_row = max_row[::-1]
    max_col = max_col[::-1]
    max_coordinates = np.vstack((max_row, max_col)).T

    # Allocate pairs
    zc_line_pairs = {}
    zc_sign_pairs = {}
    singular_points = []
    pairs_allocated = []
    row_c = []
    col_c = []
    corner = []
    zc_corner = []
    i_pair = 0
    for i_zc, zc in enumerate(zc_lines):
        if i_zc in pairs_allocated:
            continue
        # Do not pair points in the borders
        if (zc[:, 0] == wave.shape[0] - 1).any():
            pairs_allocated.append(i_zc)
            zc_corner.append(i_zc)
            row_c.append(zc[:, 0].min())
            cond = np.where(zc[:, 0].min() == zc[:, 0])[0][0]
            col_c.append(zc[cond, 1])
            continue
        if (zc[:, 1] == 0).any() or (zc[:, 1] == wave.shape[1] - 1).any():
            pairs_allocated.append(i_zc)
            zc_corner.append(i_zc)
            if (zc[:, 1] == 0).any():
                col_c.append(zc[:, 1].min())
                cond = zc[:, 1].min() == zc[:, 1]
                cond = np.where(cond)[0][0]
                row_c.append(zc[cond, 0])
            else:
                col_c.append(zc[:, 1].max())
                cond = zc[:, 1].max() == zc[:, 1]
                cond = np.where(cond)[0][0]
                row_c.append(zc[cond, 0])
            continue

        # Find closest points together through max points row direction only
        # max_row_c = max_row[i_zc]
        # max_col_c = max_col[i_zc]
        # dist = np.abs(max_col_c - max_col)
        coord = max_coordinates[i_zc]
        dist = np.sqrt(np.sum((max_coordinates - coord) ** 2, axis=1))
        dist[i_zc] = np.inf
        dist[pairs_allocated] = np.inf
        i_min = -1
        i_count = 0
        while i_min == -1:
            i_min = np.argmin(dist)
            if max_coordinates[i_min, 0] - coord[0] > 5:
                dist[i_min] = np.inf
                i_min = -1
            if i_count > 100:
                raise ValueError(f"Could not find a pair for zc {i_zc}")

        # allocate pair
        pairs_allocated.append(i_zc)
        pairs_allocated.append(i_min)
        # locate minimum column between the two points
        max_col_i_zc = max_col[i_zc]
        max_col_i_min = max_col[i_min]

        if max_col_i_zc < max_col_i_min:
            zc_line_pairs[i_pair] = [zc, zc_lines[i_min]]
            zc_sign_pairs[i_pair] = [zc_sign[i_zc], zc_sign[i_min]]
        else:
            zc_line_pairs[i_pair] = [zc_lines[i_min], zc]
            zc_sign_pairs[i_pair] = [zc_sign[i_min], zc_sign[i_zc]]
        # Set singular point as medium point between the two points
        coord = max_coordinates[i_zc]
        singular_points.append(
            np.ceil(np.mean(np.vstack((coord, max_coordinates[i_min])), axis=0))
        )
        corner.append(0)
        i_pair += 1

    # Orderd corners before pairing
    col_c = np.array(col_c)
    row_c = np.array(row_c)
    coords_c = np.vstack((row_c, col_c)).T
    # order columns
    s_col = np.argsort(col_c)
    s_col_u = np.unique(col_c[s_col])
    s_idx = []
    # order rows
    row_s = row_c[s_col]
    for i_col in s_col_u:
        if i_col == 0:
            s_col_2 = s_col[col_c[s_col] == i_col]
            row_s_2 = np.argsort(row_s[col_c[s_col] == i_col])
            s_col_2 = s_col_2[row_s_2]
            s_idx = copy.deepcopy(s_col_2)
        elif i_col == wave.shape[1] - 1:
            s_col_2 = s_col[col_c[s_col] == i_col]
            row_s_2 = np.argsort(row_s[col_c[s_col] == i_col])[::-1]
            s_col_2 = s_col_2[row_s_2]
            s_idx = np.hstack((s_idx, s_col_2))
        else:
            s_col_2 = s_col[col_c[s_col] == i_col]
            row_s_2 = np.argsort(row_s[col_c[s_col] == i_col])
            s_col_2 = s_col_2[row_s_2]
            s_idx = np.hstack((s_idx, s_col_2))

    zc_corner = np.array(zc_corner)[s_idx]
    zc_lines_corner = [zc_lines[i] for i in zc_corner]
    zc_sign_corner = [zc_sign[i] for i in zc_corner]
    max_coordinates_corner = max_coordinates[zc_corner]

    # Pair corner points
    for i_zc_2, zc in enumerate(zc_lines_corner[:-1]):
        zc_line_pairs[i_pair] = [zc, zc_lines_corner[i_zc_2 + 1]]
        zc_2 = copy.deepcopy(zc_line_pairs[i_pair])
        if zc_2[0].shape[0] > zc_2[1].shape[0]:
            zc_2[0] = zc_2[0][zc_2[1][0, 0] >= zc_2[0][:, 0]]
        elif zc_2[0].shape[0] < zc_2[1].shape[0]:
            zc_2[1] = zc_2[1][zc_2[0][0, 0] >= zc_2[1][:, 0]]
        zc = zc_2
        zc_line_pairs[i_pair] = zc
        zc_sign_pairs[i_pair] = [
            zc_sign_corner[i_zc_2],
            zc_sign_corner[i_zc_2 + 1],
        ]
        # Set singular point as medium point between the two points
        singular_points.append(
            np.ceil(
                np.mean(
                    np.vstack(
                        (
                            max_coordinates_corner[i_zc_2],
                            max_coordinates_corner[i_zc_2 + 1],
                        )
                    ),
                    axis=0,
                )
            )
        )
        corner.append(1)
        i_pair += 1

    # ============================
    # Find nesting of scales
    # ============================
    # Order by area
    area_sum = np.zeros(len(zc_line_pairs))
    masks = (
        np.zeros((len(zc_line_pairs), wave.shape[0], wave.shape[1])) * np.nan
    )
    for pair in list(zc_line_pairs.keys()):
        # Convert points within raster to 1
        zc = copy.deepcopy(zc_line_pairs[pair])
        for row in range(zc[0].shape[0]):
            zc_left = zc[0][row, 1]
            zc_right = zc[1][row, 1]
            if zc_left == zc_right:
                continue
            masks[pair, zc[0][row, 0], zc_left:zc_right] = 1
        area_sum[pair] = np.nansum(masks[pair])

    # ------------------------
    # Reorder pairs by area
    # ------------------------
    s_area = np.argsort(area_sum)[::-1]
    zc_line_pairs = {i: zc_line_pairs[s_a] for i, s_a in enumerate(s_area)}
    zc_sign_pairs = {i: zc_sign_pairs[s_a] for i, s_a in enumerate(s_area)}
    singular_points = np.array(singular_points)[s_area]
    corner = [corner[i] for i in s_area]
    pair_values = np.arange(len(zc_line_pairs))
    # remove overlapping areas
    masks_all = np.zeros((wave.shape[0], wave.shape[1])) * np.nan
    for pair in list(zc_line_pairs.keys()):
        # Convert points within raster to 1
        zc = copy.deepcopy(zc_line_pairs[pair])
        for row in range(zc[0].shape[0]):
            zc_left = zc[0][row, 1]
            zc_right = zc[1][row, 1]
            if zc_left == zc_right:
                continue
            masks_all[zc[0][row, 0], zc_left:zc_right] = pair

    # Get singular points above current pair
    s_p_r = np.array(singular_points)[:, 0] + 1
    s_p_r[s_p_r > wave.shape[0] - 1] = wave.shape[0] - 1
    s_p_new = np.vstack((s_p_r, np.array(singular_points)[:, 1])).T
    belongs_to = masks_all[s_p_new[:, 0].astype(int), s_p_new[:, 1].astype(int)]
    # ------------------------
    # Find nesting of scales
    # ------------------------
    pair = 0
    singular_points_p = singular_points[belongs_to == pair]
    pair_values_p = pair_values[belongs_to == pair]

    # order by row
    s_row = np.argsort(singular_points_p[:, 0])[::-1]
    singular_points_p = singular_points_p[s_row]
    pair_values_p = pair_values_p[s_row]

    # find peak power on cwt_matrix with the mask
    current_mask = np.zeros((wave.shape[0], wave.shape[1])) * np.nan
    current_mask[masks_all == pair] = 1
    wave_mask = np.abs(cwt_matrix) ** 2 * current_mask
    # Find peak power between pairs of singular points
    peak_row = []
    peak_col = []
    peak_pwr = []
    points_taken = []
    for i_s, s_p in enumerate(singular_points_p[:-1]):
        # Find row where singular point is located
        s_p = singular_points_p[i_s + 1]
        s_r_1 = int(s_p[0])
        # Find secon point closes to singular point
        dist = np.abs(singular_points_p[i_s + 1 :, 0] - s_p[0])
        # dist = np.sqrt(np.sum((singular_points_p[:i_s+ 1] - s_p)**2, axis=1))
        # dist[i_s] = np.inf
        i_min = -1
        iteration = 0
        while i_min == -1:
            i_min = np.argmin(dist)
            if i_min in points_taken:
                dist[i_min] = np.inf
                i_min = -1
            if iteration > 10:
                raise ValueError("Could not find second singular point")
            iteration += 1
        points_taken.append(i_min)
        points_taken.append(i_s)
        s_r_2 = int(singular_points_p[i_min][0])
        s_c = [int(s_p[1]), int(singular_points_p[i_min][1])]
        s_c_1 = np.min(s_c)
        s_c_2 = np.max(s_c)

        # Set bounds to the zc lines of the pair
        # zc_cols_1 = [zc_line_pairs[pair_values_p[i_s]][0][:, 1],
        #              zc_line_pairs[pair_values_p[i_s]][1][:, 1]]
        # zc_cols_2 = [zc_line_pairs[pair_values_p[i_min]][0][:, 1],
        #              zc_line_pairs[pair_values_p[i_min]][1][:, 1]]
        # max_zc_1 = np.max(zc_cols_1)
        # min_zc_1 = np.min(zc_cols_1)
        # max_zc_2 = np.max(zc_cols_2)
        # min_zc_2 = np.min(zc_cols_2)
        # s_c_1 = np.min([min_zc_1, min_zc_2]).astype(int)
        # s_c_2 = np.max([max_zc_1, max_zc_2]).astype(int)

        wave_clip = wave_mask[s_r_1:s_r_2, s_c_1:s_c_2]
        peak_pwr.append(np.nanmax(wave_clip))
        peak_row.append(np.where(wave_clip == peak_pwr[-1])[0][0] + s_r_1)
        peak_col.append(np.where(wave_clip == peak_pwr[-1])[1][0] + s_c_1)

    points_taken = []
    # Create root node
    i_node = 0
    tree = {}
    node_c, pairs_taken = recursive_tree_structure(
        pair, i_node, singular_points, belongs_to=belongs_to
    )

    utl.toc(time_1)

    plt.figure(figsize=(8, 5))
    plt.pcolormesh(wave_clip, cmap="Spectral", shading="auto", alpha=0.8)

    plt.figure(figsize=(8, 5))
    plt.pcolormesh(
        np.flipud(np.log2(np.abs(wave) ** 2)), cmap="Spectral", shading="auto"
    )
    plt.pcolormesh(masks_all, cmap="Spectral", shading="auto", alpha=0.8)
    plt.pcolormesh(wave_mask, cmap="Spectral", shading="auto", alpha=0.8)
    plt.scatter(peak_col, peak_row, c="k", s=10)
    # plt.scatter(max_col, max_row, c='k', s=10)
    # plot a corner
    # corner = 0
    # plt.plot(zc_lines[zc_corner[-7]][:, 1],
    #          zc_lines[zc_corner[-7]][:, 0], '--r', lw=1)
    # pair = 0
    # # plt.pcolormesh(masks[pair], cmap='Greys', shading='auto', alpha=0.3)
    # s_p_b = singular_points[belongs_to == pair]
    # # plt.scatter(s_p_b[:, 1], s_p_b[:, 0], c='k', s=10)
    # nodes = [node_c] + list(node_c.descendants)
    # for node in nodes:
    #     s_p = node.singular_point
    #     if node.parent is not None:
    #         parent_s_p = node.parent.singular_point
    #         plt.plot([s_p[1], parent_s_p[1]], [s_p[0], parent_s_p[0]], '-ok',
    #                  lw=1)

    # plt.pcolormesh(wave_mask, cmap='Greys', shading='auto', alpha=0.8)
    # plt.scatter(singular_points_p[:, 1], singular_points_p[:, 0], c='k', s=10)

    plt.plot(
        zc_line_pairs[pair][0][:, 1], zc_line_pairs[pair][0][:, 0], "-b", lw=1
    )
    plt.plot(
        zc_line_pairs[pair][1][:, 1], zc_line_pairs[pair][1][:, 0], "-r", lw=1
    )
    plt.gca().invert_yaxis()
    # for pair in list(zc_line_pairs.keys()):
    #     for i in range(2):
    #         zc_line = zc_line_pairs[pair][i]
    #         if zc_sign_pairs[pair][i] == 1:
    #             plt.plot(zc_line[:, 1], zc_line[:, 0], '-b', lw=0.5)
    #         else:
    #             plt.plot(zc_line[:, 1], zc_line[:, 0], '-r', lw=0.5)
    #         # Plot singular point
    #         plt.scatter(singular_points[pair][1], singular_points[pair][0],
    #                     color='k', s=10)
    #         # annotate singular point
    #         plt.annotate(pair, (singular_points[pair][1], singular_points[pair][0]))

    return zc_line_pairs, zc_sign_pairs, singular_points_row


def recursive_tree_structure(
    pair, i_node, singular_points, belongs_to, parent_node=None, pairs_taken=[]
):

    pairs_mask = np.where(belongs_to == pair)[0]
    # Include children
    node_c = Node(
        i_node, parent=parent_node, singular_point=singular_points[pair]
    )

    for pair_c in pairs_mask:
        if pair_c == pair:
            continue
        recursive_tree_structure(
            pair_c, i_node, singular_points, belongs_to, node_c, pairs_taken
        )

    pairs_taken.append(pair)
    i_node += 1
    return node_c, pairs_taken


def scale_space_tree(cwt_matrix):
    """
    Description:
    ------------
        Construct a ternary scale space tree from the zero crossings of the
        wavelet transform of the curvature.

        Based on Vermeulen et al. (2016) Meander tree generation

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    ------------
    :param cwt_matrix: np.ndarray,
        Wavelet transform of the curvature.
    :return:
        conn: Vector in which each index corresponds to a tree node.
            Roots nodes have a value of -1, while the other nodes have a
            value pointing to the row in conn of their parent node.
        regions: Matrix that contains the same number of indices as conn.
            for each index in conn the corresponding row in regions
            give the scale boundaries of the node and the spatial
            boundaries derived given the scale boundaries derived
            from where the zero crossing lines leave the scale space plane.
            The first column is the smallest bounding period, the second column
            is the largest bounding period, the third column is the spatial
            coordinate where the region starts and the last column is the
            spatial coordinate where the region ends.
        poly: returns a vector of cells with the same size as conn containing
            the coordinates of a polygon bounding the region of the
            corresponding node in conn. Unlike regions these coordinates follow
            the zero crossings lines bounding the region. The coordinates are
            given as [row, col].
        zc_lines: return all zero corssing lines detected. Each element in
            zc_lines contains a Nx2 matrix containing the coordinates of the
            zero crossing line given as [row, col].
        zc_sign: returns the sign of each zero crossing line in zc_lines.
    """

    # ============================
    # Flip Matrix
    # ============================
    wave = np.flipud(cwt_matrix)
    # ===========================================
    # Find zero crossing location in spectrum
    # ===========================================
    zcr = np.diff((wave > 0).astype(np.int64), n=1, axis=1)
    zcr = np.hstack((zcr, zcr[:, 0].reshape((-1, 1)) * 0))

    # Create Boundaries in the spectrum
    for row in range(zcr.shape[0]):
        row_vals = zcr[row, :]
        # Left Boundary
        if zcr[row, 0] == 0:
            try:
                zcr[row, 0] = -row_vals[row_vals != 0][0]
            except IndexError:
                zcr[row, 0] = 1
        # Right Boundary
        if zcr[row, -1] == 0:
            zcr[row, -1] = -row_vals[row_vals != 0][-1]

    # ============================
    # Find singular points
    # ============================
    # new zero-crossing lines emerge
    # which mark the edges in scale-direction of the 2d segment
    n_zeros = np.sum(np.abs(zcr), axis=1)
    # Find row (period) where singular points occur (where number of zero
    # crossings change). Note that first row is considered as singular one
    diff_m = np.hstack(([1], np.diff(n_zeros)))
    singular_points_row = np.where(diff_m > 0)[0]

    # Total number of nodes for variable initialization
    diff = np.diff(zcr[:, [0, -1]], axis=0)
    diff_m = np.vstack(([0, 0], diff)) / 2
    n_new_zc_lines_lateral = np.sum(np.abs(diff_m), axis=1)
    n_new_zc_lines = np.hstack(([0], np.diff(n_zeros)))
    n_new_zc_lines_internal = n_new_zc_lines - n_new_zc_lines_lateral
    n_new_internal_nodes = n_new_zc_lines_internal + n_new_zc_lines_internal / 2

    n_new_lateral_nodes = copy.deepcopy(n_new_zc_lines_lateral)
    n_nodes_intial = n_zeros[0] - 1
    n_nodes_per_period = n_new_internal_nodes + n_new_lateral_nodes
    n_nodes_per_period[0] = n_nodes_intial
    init_n_nodes = np.sum(n_nodes_per_period).astype(int)
    n_zc_lines = n_zeros[0] + np.sum(n_new_zc_lines)

    # =========================================================
    # Make zero-crossing lines from zero-crossing locations
    # =========================================================
    ml_idx = np.ones(wave.shape[1]) * -1
    loc_x = np.zeros(wave.shape) * np.nan

    # Initialize Variables
    xtr_colmin, xtr_colmax, xtr_rwmin, xtr_rwmax, xtr_conn = [
        np.zeros(init_n_nodes * 6) * np.nan for _ in range(5)
    ]

    xtr_poly = [0 for _ in range(init_n_nodes * 6)]
    zc_lines = [0 for _ in range(n_zc_lines * 6)]
    zc_sign = np.zeros(n_zc_lines * 6) * np.nan

    # Initialize counters
    cc = -1
    last_zc = -1
    nodes_in_per = np.zeros(zcr.shape[0])

    if init_n_nodes == 0:
        raise ValueError("No zero crossing lines found")

    # Initialize loop through every period
    for row, s_p in enumerate(singular_points_row):
        zercr_col = np.where(zcr[s_p, :] != 0)[0]
        z_new_col = np.where(np.isnan(loc_x[s_p, zercr_col]))[0]
        # ----------------------------------------------
        # Correction done by Daniel Gonzalez-Duque
        # ----------------------------------------------
        # Correct places where z_new_col has only one element
        if len(z_new_col) == 1:
            # Add the zercr_col closest to it
            dif_zercr = np.hstack(([0], np.diff(zercr_col)))
            dist = np.abs(zercr_col[z_new_col] - zercr_col)
            dist[z_new_col] = len(zcr) + 1
            argmin = np.argmin(dist)
            # Add argmin to z_new_col and sort
            z_new_col = np.hstack((z_new_col, argmin))
            z_new_col = np.sort(z_new_col)
        # ----------------------------------------------
        col_fin = z_new_col * np.nan
        for col, z_c in enumerate(z_new_col):
            # Get line through
            lrw, lcol = get_zcline(zcr, s_p, zercr_col[z_c])
            last_zc += 1
            # Flip line, because initial xwt was flipped
            zc_lines[last_zc] = np.vstack((zcr.shape[0] - lrw - 1, lcol)).T
            # zc_lines[last_zc] = np.vstack((lrw[::-1], lcol)).T
            # Capture sign of the line
            zc_sign[last_zc] = zcr[s_p, zercr_col[z_c]]
            # Column where the line ended
            col_fin[col] = lcol[-1]
            # Save  data in the matrix with the ending column
            loc_x[lrw, lcol] = lcol[-1]

        # -------------------------
        # Build the tree
        # -------------------------
        # The important variable to construct is xtr_conn as it contains
        # the connectivity of the tree. It is a vector in which each element
        # corresponds to a node (which corresponds to a 2d segment) in the tree.
        for cn, z_c in enumerate(zercr_col[:-1]):
            if (z_new_col == cn).any() | (z_new_col == cn + 1).any():
                nodes_in_per[s_p] = nodes_in_per[s_p] + 1
                cc += 1
                if cc > len(xtr_rwmin) - 1:
                    raise ValueError(
                        "Number of nodes exceeds the expected."
                        " Space scale tree cannot be constructed"
                    )
                xtr_rwmin[cc] = s_p
                xtr_colmin[cc] = loc_x[s_p, z_c]
                xtr_colmax[cc] = loc_x[s_p, zercr_col[cn + 1]]
                xtr_idx = int(
                    np.round((xtr_colmin[cc] + xtr_colmax[cc] + 1) / 2)
                )
                xtr_conn[cc] = ml_idx[xtr_idx]
                ml_idx[int(xtr_colmin[cc]) : int(xtr_colmax[cc]) + 1] = cc
                if xtr_conn[cc] != -1:
                    index = int(xtr_conn[cc])
                    xtr_rwmax[index] = s_p
                    [lline_rw, lline_col] = np.where(loc_x == xtr_colmin[index])
                    [rline_rw, rline_col] = np.where(loc_x == xtr_colmax[index])
                    lcond = (lline_rw < xtr_rwmin[index]) | (
                        lline_rw > xtr_rwmax[index]
                    )
                    rcond = (rline_rw < xtr_rwmin[index]) | (
                        rline_rw > xtr_rwmax[index]
                    )
                    lline_col = np.delete(lline_col, lcond)
                    lline_rw = np.delete(lline_rw, lcond)
                    rline_col = np.delete(rline_col, rcond)
                    rline_rw = np.delete(rline_rw, rcond)

                    rw_lines = np.hstack(
                        (lline_rw, np.flipud(rline_rw), lline_rw[0])
                    )
                    col_lines = np.hstack(
                        (lline_col, np.flipud(rline_col), lline_col[0])
                    )
                    xtr_poly[index] = np.vstack(
                        (zcr.shape[0] - rw_lines - 1, col_lines)
                    ).T

    # =========================================================
    # Correct Number of Nodes
    # =========================================================
    n_nodes = np.sum(nodes_in_per).astype(int)
    xtr_colmax = xtr_colmax[:n_nodes]
    xtr_colmin = xtr_colmin[:n_nodes]
    xtr_conn = xtr_conn[:n_nodes]
    xtr_rwmax = xtr_rwmax[:n_nodes]
    xtr_rwmin = xtr_rwmin[:n_nodes]
    xtr_poly = xtr_poly[:n_nodes]
    zc_lines = zc_lines[:n_nodes]
    zc_sign = zc_sign[:n_nodes]

    # plt.figure(figsize=(8, 5))
    # plt.pcolormesh(np.log2(np.abs(wave)**2), cmap='Spectral', shading='auto')
    # for sp in singular_points_row:
    #     plt.axhline(sp, c='k', lw=0.5, linestyle='--')
    # plt.plot(lcol, lrw, '-b', lw=1)
    # plt.scatter(xtr_colmin, xtr_rwmin, c='b', s=10)
    # plt.scatter(xtr_colmax, xtr_rwmax, c='r', s=10)

    # ======================================================================
    # Make lower bounds and polygon for leaves (smallest scale segments)
    # ======================================================================
    xtr_rwmax[np.isnan(xtr_rwmax)] = zcr.shape[0] - 1
    fleaf = np.where(xtr_rwmax == zcr.shape[0] - 1)[0]

    for cc, f_leaf in enumerate(fleaf):
        [lline_rw, lline_col] = np.where(loc_x == xtr_colmin[f_leaf])
        [rline_rw, rline_col] = np.where(loc_x == xtr_colmax[f_leaf])
        lcond = (lline_rw < xtr_rwmin[f_leaf]) | (lline_rw > xtr_rwmax[f_leaf])
        rcond = (rline_rw < xtr_rwmin[f_leaf]) | (rline_rw > xtr_rwmax[f_leaf])
        lline_col = np.delete(lline_col, lcond)
        lline_rw = np.delete(lline_rw, lcond)
        rline_col = np.delete(rline_col, rcond)
        rline_rw = np.delete(rline_rw, rcond)
        rw_lines = np.hstack((lline_rw, np.flipud(rline_rw), lline_rw[0]))
        col_lines = np.hstack((lline_col, np.flipud(rline_col), lline_col[0]))
        xtr_poly[f_leaf] = np.vstack((zcr.shape[0] - rw_lines - 1, col_lines)).T

    # ==========================
    # Output
    # ==========================
    regions = np.vstack(
        (
            zcr.shape[0] - xtr_rwmin - 1,
            zcr.shape[0] - xtr_rwmax - 1,
            xtr_colmin,
            xtr_colmax,
        )
    ).T
    return xtr_conn, regions, xtr_poly, zc_lines, zc_sign


def get_zcline(zcr, rw_start, col_start):
    """
    Description:
    ------------
        Find zero crossing line given its starting position (singular point)

        Based on Vermeulen et al. (2016) Meander tree generation

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param zcr: np.ndarray,
        zcr lines.
    :type zcr: np.ndarray
    :param rw_start: int,
        Starting row
    :type rw_start: int
    :param col_start: int,
        Starting column
    :type col_start: int
    :return: (rw, col)
    :rtype:
    """

    rw_end = zcr.shape[0] - 1
    sgn = zcr[rw_start, col_start]
    assert sgn == 1 or sgn == -1

    n_out = rw_end - rw_start + 1
    col_out = np.zeros(n_out) * np.nan
    rw_out = np.arange(rw_start, rw_end + 1)

    [rw_ss, col_ss] = np.where(zcr == sgn)

    col_out[0] = col_start

    # Make the line
    for i in range(1, n_out):
        cur_col = col_ss[rw_ss == rw_out[i]]
        col_diff = np.abs(cur_col - col_out[i - 1])
        index = np.where(col_diff == np.min(col_diff))[0][0]
        col_out[i] = cur_col[index]

    return rw_out, col_out.astype(int)


def find_peak_in_poly(poly, wave):
    """
    Description:
    ------------
        Detect peaks of a 2D function within the given simple polygons
    ____________________________________________________________________________

    Args:
    -----
    :param poly: np.ndarray,
        Polygons defining the regions of the cwt.
    :type poly: np.ndarray
    :param wave: np.ndarray,
        Wavelet transform
    :type wave: np.ndarray
    :return: (peak_pwr, peak_row, peak_col) - Peak power, row and column
    :rtype:
    """
    power = wave**2
    se_wave = sadext(wave)
    # [x -> rw_xtr, y -> col_xtr]
    [rw_xtr, col_xtr] = np.where(se_wave == 0)

    peak_pwr = np.zeros(len(poly)) * np.nan
    peak_row = np.zeros(len(poly)) * np.nan
    peak_col = np.zeros(len(poly)) * np.nan
    for i_poly, p in enumerate(poly):
        is_inside = inpolygon(p[:, 1], p[:, 0], col_xtr, rw_xtr)
        find_poly = np.where(is_inside)[0]
        if len(find_poly) == 0:
            continue
        else:
            ppeak = power[rw_xtr[find_poly], col_xtr[find_poly]]
            peak_pwr[i_poly] = np.max(ppeak)
            fmax = np.where(ppeak == peak_pwr[i_poly])[0][0]
            peak_row[i_poly] = rw_xtr[find_poly[fmax]]
            peak_col[i_poly] = col_xtr[find_poly[fmax]]

    return peak_pwr, peak_row, peak_col


def inpolygon(x, y, xv, yv):
    """
    Description:
    ------------
        Check if points are inside a polygon

        Based on Vermeulen et al. (2016) Meander tree generation

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param x: np.ndarray,
        x coordinates of polygon vertices
    :type x: np.ndarray
    :param y: np.ndarray,
        y coordinates of polygon vertices
    :type y: np.ndarray
    :param xv: np.ndarray,
        x coordinates of points to check
    :type xv:  np.ndarray,
    :param yv: np.ndarray,
        y coordinates of points to check
    :type yv: np.ndarray
    :return: is_inside: boolean vector with True if point is inside polygon
    :rtype:
    """

    # Define vertices of polygon
    poly_verts = np.vstack((x, y)).T

    # Create path from vertices
    path = Path(poly_verts)

    # Check if points are inside polygon
    points = np.vstack((xv, yv)).T
    is_inside = path.contains_points(points, radius=0.5)
    return is_inside


def hexl(inmat):
    """
    Description:
    ------------
        Constructs a hexagonal lattice for saddle points and extremes detection.

        The size of the lattixe is [inmat.size(), 6], i.e. it has six elements
        in the third dimension each containing one of the dix surrounding
        elements in the lattice, in the following order: top left, top center,
        mid left, mid right, bottom center, bottom right.

        This function was coded in MATLAB initially on Vermulen et al. (2016),
        and is based on Kuijper (2004).

        References:
        
        Kuijper, A. (2004), On detecting all saddle points in 2D images, Pattern
           Recogn. Lett., 25 (15), 1665-1672, doi:10.1016/j.patrec.2004.06.017

        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.

    ____________________________________________________________________________

    Args:
    -----
    :param inmat: np.ndarray,
        Input matrix
    :type inmat:  np.ndarray
    :return:
    :rtype:
    """
    nr, nc = inmat.shape
    assert nr > 2 and nc > 2

    lat1, lat2, lat3, lat4, lat5, lat6 = (
        inmat,
        inmat,
        inmat,
        inmat,
        inmat,
        inmat,
    )
    lat1 = np.delete(
        np.delete(lat1, [nc - 1, nc - 2], axis=1), [nr - 1, nr - 2], axis=0
    )  # top left
    lat2 = np.delete(
        np.delete(lat2, [nc - 1, nc - 2], axis=1), [0, nr - 1], axis=0
    )  # top center
    lat3 = np.delete(
        np.delete(lat3, [0, nc - 1], axis=1), [nr - 1, nr - 2], axis=0
    )  # mid left
    lat4 = np.delete(
        np.delete(lat4, [0, nc - 1], axis=1), [0, 1], axis=0
    )  # mid right
    lat5 = np.delete(
        np.delete(lat5, [0, nr - 1], axis=0), [0, 1], axis=1
    )  # bottom center
    lat6 = np.delete(
        np.delete(lat6, [0, 1], axis=0), [0, 1], axis=1
    )  # bottom right
    lat = np.concatenate(
        (
            lat1[:, :, np.newaxis],
            lat2[:, :, np.newaxis],
            lat4[:, :, np.newaxis],
            lat6[:, :, np.newaxis],
            lat5[:, :, np.newaxis],
            lat3[:, :, np.newaxis],
        ),
        axis=2,
    )  # Concatenate the six arrays
    return lat


def sadext(inmat):
    """
    Description:
    ------------
        Detect saddle points and extremes in a 2D matrix (Vermulen, 2016;
        Kuijper, 2004)

        ID has the same size as inmat and contains an id which can have one
        of the following values:

        0: The point is a local extreme
        2: the point is a regular point
        4: The point is a saddle point
        6: The point is a degenerate saddle point

        This function requires the hexl function to build the hexagonal lattice.

        References:
        
        Kuijper, A. (2004), On detecting all saddle points in 2D images, Pattern
        Recogn. Lett., 25 (15), 1665-1672, doi:10.1016/j.patrec.2004.06.017

        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param inmat:
    :type inmat:
    :return:
    :rtype:
    """
    inmat6 = hexl(inmat)
    ccell = copy.deepcopy(inmat)
    ccell = np.delete(ccell, [0, -1], axis=0)
    ccell = np.delete(ccell, [0, -1], axis=1)
    # stack input matrix with itself and its first column
    inmat6_2 = np.dstack((inmat6, inmat6[:, :, 0]))
    # broadcast comparison between inmat6 and ccell
    greater_than_ccell = np.greater(
        inmat6_2, ccell.reshape((ccell.shape[0], ccell.shape[1], -1))
    ).astype(int)
    # absolute difference between adjacent columns along the third axis
    diff_greater_than_ccell = np.abs(np.diff(greater_than_ccell, axis=2))
    # compute id
    id_value = np.sum(diff_greater_than_ccell, axis=2) / 2
    # create a boolean mask to set id as nan wherever any element
    # or its surrounding lattice is nan
    nan_index = np.any(np.isnan(np.dstack((inmat6, ccell))), axis=2)
    id_value[nan_index] = np.nan
    # duplicate first and last column
    id_value = np.concatenate(
        (id_value[:, [0]], id_value, id_value[:, [-1]]), axis=1
    )
    # duplicate first and last row
    id_value = np.concatenate(
        (id_value[[0], :], id_value, id_value[[-1], :]), axis=0
    )

    return id_value


def remove_nodes(conn, frm):
    """
    Description:
    ------------
        Remove nodes from a tree (they become isolated root nodes).

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :param frm: np.ndarray,
        Vector with nodes to remove
    :type frm: np.ndarray
    :return: conn: Connection matrix
    :rtype:
    """
    conn = copy.deepcopy(conn)
    conn = check_conn(conn)

    for cn, f in enumerate(frm):
        conn[conn == f] = conn[f]
        conn[f] = -1

    return conn


def remove_peak_from_nodes(frm, peak_row, peak_col, peak_pwr):
    """
    Description:
    ------------
        Remove a peak from the nodes.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param frm: np.ndarray,
        Vector with nodes to remove
    :type frm: np.ndarray
    :param peak_row: np.ndarray,
        Row indices of the peaks
    :type peak_row: np.ndarray
    :param peak_col: np.ndarray,
        Column indices of the peaks
    :type peak_col: np.ndarray
    :param peak_pwr: np.ndarray,
        Power of the peaks
    :type peak_pwr: np.ndarray
    :return: peak_row, peak_col, peak_pwr: np.ndarray,
        Row and column indices and power of the peaks
    """
    frm = np.array(frm)
    peak_row[frm] = np.nan
    peak_col[frm] = np.nan
    peak_pwr[frm] = np.nan
    return peak_row, peak_col, peak_pwr


def detect_meanders(wave, conn, peak_row, peak_col):
    """
    Description:
    ------------
        Detect meanders in a wavelet transform from generated tree.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param wave: np.ndarray,
        Wavelet transform
    :type wave: np.ndarray
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :param peak_row: np.ndarray,
        Row indices of the peaks
    :type peak_row: np.ndarray
    :param peak_col: np.ndarray,
        Column indices of the peaks
    :type peak_col: np.ndarray
    :return:
    :rtype:
    """
    pwr = wave**2
    # Check input
    conn = check_conn(conn)
    # peak_row = peak_row
    # peak_col = peak_col

    ml_out = []
    ml_iter = True
    idx_out = []
    while ml_iter:
        idx_out = []
        ml_iter = False
        for cp, c in enumerate(conn):
            if c == -1:
                continue
            if np.any(conn == cp):
                continue
            branch = get_branch(cp, conn)
            branch = np.setdiff1d(branch, ml_out)
            if len(branch) == 0:
                continue

            if np.any(np.isnan(peak_row[branch])):
                raise ValueError(
                    "NaN values in peak_row, review you conn matrix"
                )
            rw_indices = peak_row[branch].astype(int)
            col_indices = peak_col[branch].astype(int)

            br_pwr = pwr[rw_indices, col_indices]
            br_m_id = np.argmax(br_pwr)
            max_idx = branch[br_m_id]
            if max_idx is not None and max_idx not in idx_out:
                idx_out.append(max_idx)
            if br_m_id > 0:
                new_ml = branch[:br_m_id]
                ml_out += list(new_ml)
                ml_iter = True
    idx_out = np.array(idx_out)
    return idx_out


def meander_bounds(poly, meander_id, peak_row, include_all=False):
    """
    Description:
    ------------
        Computes the start and end-point of meander

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param poly: np.ndarray,
        Polygon in the wavelet
    :param meander_id: np.ndarray,
        Meander indices
    :param peak_row: np.ndarray,
        Row indices of the peaks
    :return:
        bounds: bounding indices for the meander
    """

    m_row = peak_row[meander_id]
    m_poly = [poly[i] for i in meander_id]
    bounds = np.zeros((len(meander_id), 2), dtype=int)
    for c_pol in range(len(m_poly)):
        cond = m_poly[c_pol][m_poly[c_pol][:, 0] == m_row[c_pol], 1]
        try:
            bounds[c_pol, :] = np.unique(cond)
        except:
            if include_all:
                bounds[c_pol, :] = [0, 0]
            else:
                try:
                    meander_id.pop(c_pol)
                except AttributeError:
                    try:
                        meander_id = np.delete(meander_id, c_pol)
                    except IndexError:
                        continue
            continue
    return bounds


def clean_tree(conn, meander_id):
    """
    Description:
    ------------
        Remove all nodes from the tree with scales smaller than the meander
        scale.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :param meander_id: np.ndarray,
        Meander indices
    :type meander_id: np.ndarray
    :return:
    :rtype:
    """
    conn = check_conn(conn)

    frm = []
    for cp, c in enumerate(conn):
        if c == -1:
            continue
        if cp in meander_id:
            continue
        # Root to leaf
        branch = np.flipud(get_branch(cp, conn))
        m_node = np.intersect1d(branch, meander_id)
        if len(m_node) > 0:
            m_node_idx = np.where(branch == m_node[0])[0][0]
            frm = np.concatenate(
                (frm, np.setdiff1d(branch[m_node_idx + 1 :], frm))
            )

    if len(frm) > 0:
        conn = remove_nodes(conn, frm.astype(int))
    return conn


def get_branch(cp, conn):
    """
    Description:
    ------------
        Get the branch of a node in a tree from leafs to root.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param cp: int,
        Node index
    :type cp: int
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :return:
    :rtype:
    """
    branch_idx = []
    while conn[cp] != -1:
        branch_idx.append(cp)
        cp = conn[cp]
    branch_idx.append(cp)
    return np.array(branch_idx)


def get_centers(
    conn,
    peak_row,
    peak_col,
    period,
    ds,
    x,
    y,
    extract_all=False,
    bound_to_poly=False,
    bounds=None,
):
    """
    Description:
    ------------
        Computes the center of curving sections in the multiple loop tree

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :param peak_row: np.ndarray,
        Row indices of the peaks
    :type peak_row: np.ndarray
    :param peak_col: np.ndarray,
        Column indices of the peaks
    :type peak_col: np.ndarray
    :param period: np.ndarray,
        Period of the wavelet transform
    :type period: np.ndarray
    :param ds: float,
        Delta of distance
    :type ds: float
    :param x: np.ndarray,
        X coordinates in the planimetry
    :type x: np.ndarray
    :param y: np.ndarray,
        Y coordinates in the planimetry
    :type y: np.ndarray
    :return: xc: np.ndarray, X coordinates of the center of curving sections
             yc: np.ndarray, Y coordinates of the center of curving sections
    :rtype: np.ndarray
    """
    conn = check_conn(conn)

    x_c = np.zeros_like(conn) * np.nan
    y_c = np.zeros_like(conn) * np.nan
    half_periods = np.zeros_like(conn) * np.nan
    x_1_all = np.zeros_like(conn) * np.nan
    x_2_all = np.zeros_like(conn) * np.nan
    x_3_all = np.zeros_like(conn) * np.nan
    y_1_all = np.zeros_like(conn) * np.nan
    y_2_all = np.zeros_like(conn) * np.nan
    y_3_all = np.zeros_like(conn) * np.nan
    cc_x = np.zeros_like(conn) * np.nan
    cc_y = np.zeros_like(conn) * np.nan
    r_x = np.zeros_like(conn) * np.nan
    r_y = np.zeros_like(conn) * np.nan
    beg_idx_all = np.zeros_like(conn) * np.nan
    end_idx_all = np.zeros_like(conn) * np.nan
    mid_idx_all = np.zeros_like(conn) * np.nan
    fgood = np.where(~np.isnan(peak_row))[0]
    if bound_to_poly:
        bounds = bounds
    else:
        bounds = None

    for i_cn, cn in enumerate(fgood):
        per = period[int(peak_row[cn])] / 2
        span = round(per / 4 / ds)
        beg_idx = max(0, int(peak_col[cn]) - span)
        end_idx = min(x.size, int(peak_col[cn]) + span)
        if beg_idx < 0:
            beg_idx = 0
        if end_idx > len(x) - 1:
            end_idx = len(x) - 1
        mid_idx = (end_idx + beg_idx) // 2

        if bounds is not None:
            idx_st = bounds[cn][0]
            idx_end = bounds[cn][1]
            dif_idx = idx_end - idx_st
            dif_idx_span = end_idx - beg_idx
            if dif_idx > dif_idx_span:
                beg_idx = bounds[cn][0]
                end_idx = bounds[cn][1]
                mid_idx = int(peak_col[cn])

        mid_idx = (end_idx + beg_idx) // 2
        x1, y1 = x[beg_idx], y[beg_idx]
        x2, y2 = x[mid_idx], y[mid_idx]
        x3, y3 = x[end_idx], y[end_idx]
        try:
            tri = Delaunay(np.array([[x1, y1], [x2, y2], [x3, y3]]))
            cc = gf.circumcenter(tri)
            x_c[cn], y_c[cn] = cc[0], cc[1]
        except:
            # print('Error in Delaunay triangulation, found colinear points')
            continue
        try:
            coordinates = np.vstack((x[beg_idx:end_idx], y[beg_idx:end_idx])).T
            x_c[cn], y_c[cn], r, sigma = taubinSVD(coordinates)
        except:
            continue
        l = euclidean([x2, y2], [x_c[cn], y_c[cn]])
        rvec = np.array([x_c[cn] - x2, y_c[cn] - y2]) / l
        # Compute xi and yi coordinates as a radius of the half period distance
        #   along the vector pointing to the in-center.
        # x_c[cn] = x2 + rvec[0] * per / np.pi
        # y_c[cn] = y2 + rvec[1] * per / np.pi
        x_c[cn] = x2 + rvec[0] * per / (2 * np.pi)
        y_c[cn] = y2 + rvec[1] * per / (2 * np.pi)
        # Store additional information
        r_x[cn] = rvec[0]
        r_y[cn] = rvec[1]
        half_periods[cn] = per
        x_1_all[cn] = x1
        x_2_all[cn] = x2
        x_3_all[cn] = x3
        y_1_all[cn] = y1
        y_2_all[cn] = y2
        y_3_all[cn] = y3
        cc_x[cn] = cc[0]
        cc_y[cn] = cc[0]
        beg_idx_all[cn] = beg_idx
        end_idx_all[cn] = end_idx
        mid_idx_all[cn] = mid_idx
    if extract_all:
        vars_to_return = [
            x_c,
            y_c,
            half_periods,
            r_x,
            r_y,
            x_1_all,
            x_2_all,
            x_3_all,
            y_1_all,
            y_2_all,
            y_3_all,
            cc_x,
            cc_y,
            beg_idx_all,
            end_idx_all,
            mid_idx_all,
        ]
        return vars_to_return
    else:
        return x_c, y_c


def get_tree_scales_dict(
    conn,
    peak_row,
    peak_col,
    peak_pwr,
    wave,
    wavelength,
    scales,
    ds,
    x,
    y,
    s_curvature,
    poly,
    include_metrics=True,
    bound_to_poly=False,
):
    """
    Description:
    ------------
        Computes the scales of the tree branches and collects all
        the information in a dictionary.
    ____________________________________________________________________________

    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :param peak_row: np.ndarray,
        Row indices of the peaks
    :type peak_row: np.ndarray
    :param peak_col: np.ndarray,
        Column indices of the peaks
    :type peak_col: np.ndarray
    :param peak_pwr: np.ndarray,
        Power of the peaks
    :type peak_pwr: np.ndarray
    :param wavelength: np.ndarray,
        Period of the peaks
    :type wavelength: np.ndarray
    :param scales: np.ndarray,
        Scales of the peaks
    :type scales: np.ndarray
    :param ds: float,
        Sampling distance
    :type ds: float
    :param x: np.ndarray,
        X coordinates of the river
    :type x: np.ndarray
    :param y: np.ndarray,
        Y coordinates of the river
    :type y: np.ndarray
    :param poly: np.ndarray,
        Polygon deliniating the scale tree
    :type poly: np.ndarray


    """
    conn = check_conn(conn)

    branch_id = 0
    conn_unique, idx_conn_u = np.unique(conn, return_index=True)
    conn_unique = conn_unique[::-1][:-1]
    idx_conn_u = idx_conn_u[::-1][:-1]

    idx_sort = np.argsort(peak_pwr)[::-1]
    conn_sorted = conn[idx_sort]
    idx_conn_u = idx_sort[conn_sorted != -1]
    # Create the dictionary
    var_labels = [
        "branch_id",
        "idx_conn",
        "branch",
        "conn",
        "levels_root_leaf",
        "levels_leaf_root",
        "link_branch_by_level",
        "peak_col",
        "peak_row",
        "peak_pwr",
        "s_c",
        "scales_c",
        "wavelength_c",
        "x_c",
        "y_c",
        "r_x",
        "r_y",
        "x_1",
        "x_2",
        "x_3",
        "y_1",
        "y_2",
        "y_3",
        "cc_x",
        "cc_y",
        "beg_idx",
        "end_idx",
        "mid_idx",
        "radius",
        "idx_planimetry_start",
        "idx_planimetry_end",
        "sn",
        "l",
        "lambda_value",
        "fl",
        "sk",
        "meander_in_level_root_leaf",
    ]
    tree_scales = {i: [] for i in var_labels}

    ml_out = []
    for cp in idx_conn_u:
        c = conn[cp]
        if c == -1:
            continue
        # get branch (order leafs to root)
        branch = get_branch(cp, conn)
        peak_col_branch = peak_col[branch]
        # Correct branch
        branch = branch[~np.isnan(peak_col_branch)]
        # Do not iterate over branches already processed
        branch_check = np.setdiff1d(branch, ml_out)
        if len(branch_check) == 0:
            continue

        # Inlcude Branch
        tree_scales["branch_id"].append(branch_id)
        tree_scales["idx_conn"].append(branch)
        tree_scales["conn"].append(conn[branch])

        # Add Levels
        # order from root to leaf
        level = np.arange(len(branch))
        tree_scales["levels_leaf_root"].append(level)
        tree_scales["levels_root_leaf"].append(level[::-1])
        tree_scales["meander_in_level_root_leaf"].append(level[::-1][0])

        # Add Peaks in CWT
        tree_scales["peak_col"].append(peak_col[branch])
        tree_scales["peak_row"].append(peak_row[branch])
        tree_scales["peak_pwr"].append(peak_pwr[branch])

        # Add location in wavelet plot
        s_c = s_curvature[peak_col[branch].astype(int)]
        wavelength_c = wavelength[peak_row[branch].astype(int)]
        scales_c = scales[peak_row[branch].astype(int)]
        tree_scales["s_c"].append(s_c)
        tree_scales["scales_c"].append(scales_c)
        tree_scales["wavelength_c"].append(wavelength_c)

        # Add planimetry information
        conn_branch = conn[branch]
        peak_row_branch = peak_row[branch]
        peak_col_branch = peak_col[branch]

        # Extract bounds
        bounds = meander_bounds(poly, branch, peak_row)
        tree_scales["idx_planimetry_start"].append(bounds[:, 0])
        tree_scales["idx_planimetry_end"].append(bounds[:, 1])

        # fgood = np.where(~np.isnan(peak_row))[0]
        # print(branch, peak_row[branch], fgood[branch], fgood[conn])
        vars_to_return = get_centers(
            conn_branch,
            peak_row_branch,
            peak_col_branch,
            wavelength,
            ds,
            x,
            y,
            extract_all=True,
            bound_to_poly=bound_to_poly,
            bounds=bounds,
        )

        tree_scales["x_c"].append(vars_to_return[0])
        tree_scales["y_c"].append(vars_to_return[1])
        tree_scales["r_x"].append(vars_to_return[3])
        tree_scales["r_y"].append(vars_to_return[4])
        tree_scales["x_1"].append(vars_to_return[5])
        tree_scales["x_2"].append(vars_to_return[6])
        tree_scales["x_3"].append(vars_to_return[7])
        tree_scales["y_1"].append(vars_to_return[8])
        tree_scales["y_2"].append(vars_to_return[9])
        tree_scales["y_3"].append(vars_to_return[10])
        tree_scales["cc_x"].append(vars_to_return[11])
        tree_scales["cc_y"].append(vars_to_return[12])
        tree_scales["beg_idx"].append(vars_to_return[13])
        tree_scales["end_idx"].append(vars_to_return[14])
        tree_scales["mid_idx"].append(vars_to_return[15])

        # Add radius
        x_2 = vars_to_return[6]
        y_2 = vars_to_return[9]
        r_x = vars_to_return[3]
        r_y = vars_to_return[4]

        x_center = r_x + x_2
        y_center = r_y + y_2
        x_c = vars_to_return[0]
        y_c = vars_to_return[1]
        radius_x = np.array([x_center, x_c])
        radius_y = np.array([y_center, y_c])
        dist_r = np.sqrt((x_center - x_c) ** 2 + (y_center - y_c) ** 2)
        tree_scales["radius"].append(dist_r)

        if include_metrics:
            # Additional metrics
            l = []
            lambda_values = []
            sn_values = []
            for b in bounds:
                x_s = x[b[0] : b[1] + 1]
                y_s = y[b[0] : b[1] + 1]

                # Distances
                l.append(RF.calculate_l(x_s, y_s))
                lambda_values.append(RF.calculate_lambda(x_s, y_s))
                # Sinuosity
                sn_values.append(
                    RF.calculate_sinuosity(l[-1], lambda_values[-1])
                )

            # Store data
            tree_scales["sn"].append(np.array(sn_values))
            tree_scales["l"].append(np.array(l))
            tree_scales["lambda_value"].append(np.array(lambda_values))

            # Calculate flatness and skewness
            ids = np.arange(len(bounds[:, 0]))
            scale = scales
            peak_row_branch = peak_row[branch]
            peak_col_branch = peak_col[branch]
            sk_val, fl_val = calculate_meander_shape(
                wave,
                wavelength,
                peak_row_branch,
                peak_col_branch,
                ids,
                bounds,
                scale,
                ds,
            )
            tree_scales["sk"].append(sk_val)
            tree_scales["fl"].append(fl_val)

        # Add branch to already processed
        ml_out += list(branch)
        branch_id += 1

    # Calculate general level
    max_level = np.max([np.max(l) for l in tree_scales["levels_root_leaf"]])
    tree_scales["general_levels"] = np.arange(max_level + 1)
    tree_scales["max_level_in_branch"] = np.array(
        [np.max(l) for l in tree_scales["levels_root_leaf"]]
    )

    # Add Linking
    branches = tree_scales["idx_conn"]
    rows = len(tree_scales["branch_id"])
    cols = np.max([len(b) for b in branches])
    branches_array = np.zeros((rows, cols)) * np.nan
    conn_array = np.zeros((rows, cols)) * np.nan
    # Fill the branches array
    for i, b in enumerate(branches):
        branches_array[i, : len(b)] = b
        conn_array[i, : len(b)] = conn[b]

    # Cycle through the branch_ids
    for branch_id in tree_scales["branch_id"]:
        b = branches_array[branch_id]
        # cycle through levels in root_leaf order
        levels = tree_scales["levels_root_leaf"][branch_id]
        link_branch_by_level = []
        for i_l in range(len(levels)):
            b_l = b[i_l]
            idx_b = np.where(branches_array == b_l)[0].astype(int)
            link_branch_by_level.append(idx_b)
        tree_scales["link_branch_by_level"].append(link_branch_by_level[::-1])

    return tree_scales


def n_child(conn):
    """
    Description:
    ------------
        Get the number of children for each node.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :return:
    :rtype:
    """
    # Check connection matrix
    conn = copy.deepcopy(conn)
    conn = check_conn(conn)
    # conn[conn == -1] = 0
    conn += 1
    # Count children for each node in tree
    nc = np.bincount(conn, minlength=conn.shape[0] + 1)
    return nc[1:]


def check_conn(conn):
    """
    Description:
    ------------
        Check if the connection matrix is valid.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________


    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix
    :type conn: np.ndarray
    :return: conn
    :rtype:
    """
    assert isinstance(conn, np.ndarray)
    assert np.isinf(conn).sum() == 0
    assert (conn >= -1).all()
    if conn.dtype != "int":
        conn = conn.astype(int)

    return conn


def plot_regions(regions, ax=None, **kwargs):
    """
    Description:
    ------------
        Plot tree regions as done by Witkin 1984 based on the matlab code
        by Vermulen et al. (2016)

        References:
        
        Witkin, A. P. (1984), Scale-space filtering: A new approach to
        multi-scale description, in Acoustics, Speech, and Signal Processing,
        IEEE International Conference on ICASSP ’84, vol. 9, pp. 150–153,
        IEEE, doi:10.1109/ICASSP.1984.1172729.

        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.

    ____________________________________________________________________________

    Args:
    -----
    :param regions: np.ndarray,
        Regions array comming from the scale_space_tree function.
    :type regions: np.ndarray
    :param ax: axis handle,
        Axis handle to plot the regions.
    :type ax: plt.axis
    :param kwargs: dict,
        Keyword arguments to pass to the plot function.
    :type kwargs: dict
    :return: h
    :rtype:
    """
    if ax is None:
        ax = plt.gca()
    columns_1 = np.array([2, 2, 3, 3, 2])
    columns_2 = np.array([0, 1, 1, 0, 0])
    h = ax.plot(regions[:, columns_1].T, regions[:, columns_2].T, **kwargs)
    return h


def plot_tree(conn, x, y, ax=None, **kwargs):
    """
    Description:
    ------------
        Plot a tree given its connection matrix and its node coordinates.

        This function was coded in MATLAB initially on Vermulen et al. (2016).

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.
    ____________________________________________________________________________

    Args:
    -----
    :param conn: np.ndarray,
        Connection matrix.
    :type conn: np.ndarray
    :param x: np.ndarray,
        x node coordinates.
    :type x: np.ndarray
    :param y: np.ndarray,
        y node coordinates.
    :type y: np.ndarray
    :param ax: axis handle,
        Axis handle to plot the tree.
    :type ax: plt.axis
    :param kwargs:
    :type kwargs:
    :return:
    :rtype:
    """
    if ax is None:
        ax = plt.gca()
    conn = check_conn(conn)
    assert x.shape == y.shape
    assert conn.shape == y.shape
    nc = n_child(conn)
    h = [0 for _ in range(conn.shape[0])]
    for cp, c_n in enumerate(conn):
        if c_n == -1:
            # continue
            if nc[cp] == 0:
                # Plot root node
                h[cp] = ax.plot(x[cp], y[cp], **kwargs)
            else:
                continue
        else:
            # Plot connection
            h[cp] = ax.plot([x[cp], x[c_n]], [y[cp], y[c_n]], **kwargs)
            ax.annotate(str(cp), (x[cp], y[cp]))
    return h


def calculate_meander_shape(
    wave, wavelength, peak_row, peak_col, meander_id, bounds, scale, ds
):
    """
    Description:
    ------------
        Calculate the meander flatness and skewness using equation (7) and (8)
        from Vermulen et al. (2016).

        This function is based on the matlab code by Vermulen et al. (2016)

        References:
        
        Vermeulen, B., A. J. F. Hoitink, G. Zolezzi, J. D. Abad, and R. Aalto
        (2016), Multi-scale structure of meanders, Geophys. Res. Lett., 43,
        doi:10.1002/2016GL068238.

    Args:
    ------
        :param wave: np.ndarray,
            Waveform array.
        :param wavelength: np.ndarray,
            Period array.
        :param peak_row: np.ndarray,
            Peak row array.
        :param peak_col: np.ndarray,
            Peak column array.
        :param meander_id: np.ndarray,
            Meander id array.
        :param bounds: np.ndarray,
            Bounds array.
        :param scale: np.ndarray,
            Scale array.
        :param ds: float,
            Sampling distance.
        :return: sk_val, fl_val
    """

    sk_val = np.zeros(len(meander_id)) * np.nan
    fl_val = np.zeros(len(meander_id)) * np.nan

    for cm, id_m in enumerate(meander_id):
        m_filt = np.arange(bounds[cm, 0], bounds[cm, 1] + 1)
        mp_filt = int(peak_row[id_m])
        if m_filt[-1] - m_filt[0] == 0:
            continue
        phi = (m_filt - m_filt[0]) / (m_filt[-1] - m_filt[0]) * np.pi
        m_per = wavelength[int(mp_filt)]
        # Filter to select scale of shape parameters (1/3 of meander scale)
        sp_filt = np.diff((wavelength - m_per / 3) > 0, prepend=0) == 1
        sper = wavelength[sp_filt]
        if len(sper) == 0:
            continue

        # Spectrum at meander scale
        sign = np.sign(wave[int(peak_row[id_m]), int(peak_col[id_m])])
        m_wave = wave[mp_filt, m_filt] * sign
        # Spectrum at shape scale
        s_wave = wave[sp_filt, m_filt] * sign

        # Detect Peaks
        try:
            s_peak_idx = (
                np.where(np.diff(np.diff(s_wave) > 0, prepend=0) == -1)[0] + 1
            )
            s_peak_max = np.max(s_wave[s_peak_idx])
        except ValueError:
            s_peak_max = 0

        m_peak_max = np.max(m_wave)
        scaling = np.sqrt(ds / scale[sp_filt])
        skv = np.cos(3 * phi)
        fatv = np.sin(3 * phi)

        sk_val[cm] = (
            scaling
            * np.sum(s_wave * skv)
            / np.sum(skv**2)
            * s_peak_max
            / m_peak_max
        )
        fl_val[cm] = (
            scaling
            * np.sum(s_wave * fatv)
            / np.sum(fatv**2)
            * s_peak_max
            / m_peak_max
        )

    return sk_val, fl_val
