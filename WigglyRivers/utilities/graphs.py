# -*- coding: utf-8 -*-
# ______________________________________________________________________________
# ______________________________________________________________________________
#
#                       Coded by Daniel González Duque
#                           Last revised 2023-07-24
# ______________________________________________________________________________
# ______________________________________________________________________________
"""
The functions here are used to plot Meanders
"""
# ------------------------
# Importing Modules
# ------------------------
import copy
from typing import Union, Tuple
import pathlib as pl

# Data Managment
import numpy as np
from anytree import Node

# Graphs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import pyproj
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Personal libraries
from . import utilities as utl
from .classExceptions import *
from ..wavelet_tree import WaveletTreeFunctions as WTFunc
from ..rivers import RiverFunctions as RF

# from ..rivers import RiverTransect
# from ..rivers import RiverDatasets


# ------------------------
# Functions
# ------------------------
class MidPointLogNorm(colors.LogNorm):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        colors.LogNorm.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        x, y = [np.log(self.vmin), np.log(self.midpoint), np.log(self.vmax)], [
            0,
            0.5,
            1,
        ]
        return np.ma.array(
            np.interp(np.log(value), x, y), mask=result.mask, copy=False
        )


class MidPointNorm(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        colors.Normalize.__init__(self, vmin=vmin, vmax=vmax, clip=clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        normalized_min = max(
            0,
            1
            / 2
            * (
                1
                - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))
            ),
        )
        normalized_max = min(
            1,
            1
            / 2
            * (
                1
                + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))
            ),
        )
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [
            normalized_min,
            normalized_mid,
            normalized_max,
        ]
        return np.ma.masked_array(np.interp(value, x, y))


def plot_wavelet_system(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray,
    s_curvature: np.ndarray,
    cwt_matrix: np.ndarray,
    scales: np.ndarray,
    cwt_period: np.ndarray,
    zc_lines: Union[np.ndarray, None] = None,
    zc_sign: Union[np.ndarray, None] = None,
    poly: Union[np.ndarray, None] = None,
    ml_tree: Union[np.ndarray, None] = None,
    peak_row: Union[np.ndarray, None] = None,
    peak_col: Union[np.ndarray, None] = None,
    xc: Union[np.ndarray, None] = None,
    yc: Union[np.ndarray, None] = None,
    regions: Union[np.ndarray, list, None] = None,
    save: bool = False,
    path: Union[str, pl.Path, None] = None,
    name: Union[str, None] = None,
    cmap: Union[str] = "Spectral",
    meanders: Union[np.ndarray, list, None] = None,
    curvature_side: int = 1,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """This function plots the river, the curvature, and the wavelet response.
    It will also plot the zero-crossings and the peaks of the wavelet response,
    and the tree that shows the location of the meanders with dots.

    Args:
        x (np.ndarray): x coordinates of the river.
        y (np.ndarray): y coordinates of the river.
        c (np.ndarray): Curvature of the river.
        s_curvature (np.ndarray): Arc length of the river.
        cwt_matrix (np.ndarray): Wavelet response of the river.
        scales (np.ndarray): Scales of the wavelet response.
        cwt_period (np.ndarray): Period of the wavelet response.
        zc_lines (Union[np.ndarray, None], optional): zero-crossing lines of
            the wavelet response. If None this lines will not be plotted.
            Defaults to None.
        zc_sign (Union[np.ndarray, None], optional): sign of the zero-crossing
            lines of the wavelet response. If None this lines will not be
            plotted. Defaults to None.
        poly (Union[np.ndarray, None], optional): Polygons that delimits the
            region in the wavelet response. If None this regions will not be
            plotted. Defaults to None.
        ml_tree (Union[np.ndarray, None], optional): Connection between the
            nodes of the tree. If None this nodes will not be plotted.
            Defaults to None.
        peak_row (Union[np.ndarray, None], optional): Row of the peaks of the
            wavelet response for each polygon region. Defaults to None.
        peak_col (Union[np.ndarray, None], optional): Columns of the peaks of
            the wavelet response for each plygon region. Defaults to None.
        xc (Union[np.ndarray, None], optional): x coordinates of the tree nodes
            projected to the planimetry. Defaults to None.
        yc (Union[np.ndarray, None], optional): y coordinated of the tree nodes
            projected to the planimetry. Defaults to None.
        regions (Union[np.ndarray, list, None], optional): regions of the
            wavelet response. Defaults to None.
        save (bool, optional): flag to save the figure. If False the figure will
            not be saved. Defaults to False.
        path (Union[str, pl.Path, None], optional): path to save the figure.
            This parameter will only be used if save is True. Defaults to None.
        name (Union[str, None], optional): name of the figure.
            This parameter will only be used if save is True. Defaults to None.
        cmap (Union[str], optional): color map of the wavelet response function.
            Defaults to "Spectral".
        meanders (Union[np.ndarray, list, None], optional): meander x and y
            coordinates. Defaults to None.
        curvature_side (int, optional): side of the curvature to plot. This
            parameter will only be used if meanders is not None. The parameter
            can be 1 or -1. Defaults to 1.

    Returns:
        Tuple[plt.Figure, plt.Axes]: figure and axes of the plot.
    """
    # Prepare cwt_data
    wave = np.log2(cwt_matrix**2)
    # wave = cwt_matrix
    # Plot the tree
    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    # River
    ax[0].plot(x, y, "-k")
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    # Add meanders
    if meanders is not None:
        for id_m in list(meanders.keys()):
            x_m = meanders[id_m].x
            y_m = meanders[id_m].y
            c_side = meanders[id_m].curvature_side
            if curvature_side != c_side:
                continue
            mid_x = x_m[len(x_m) // 2]
            mid_y = y_m[len(y_m) // 2]
            if meanders[id_m].curvature_side == 1:
                color_m = "r"
                linestyle = "-"
            elif meanders[id_m].curvature_side == -1:
                color_m = "b"
                linestyle = "--"
            ax[0].plot(x_m, y_m, color=color_m, linestyle=linestyle)
            ax[0].scatter(x_m[0], y_m[0], c=color_m, s=30, marker="*")
            ax[0].scatter(x_m[-1], y_m[-1], c=color_m, s=30, marker="^")
            ax[0].annotate(str(id_m), xy=(mid_x, mid_y), fontsize=14)
            # ax[0].text(mid_x, mid_y, str(id_m), fontsize=16)

    # Add planimetry tree
    if xc is not None:
        WTFunc.plot_tree(ml_tree, xc, yc, ax=ax[0], **kwargs)
    # Curvature
    ax[1].plot(s_curvature, c, "-k")
    ax[1].set_xlim([np.min(s_curvature), np.max(s_curvature)])
    ax[1].set_xlabel("Distance (m)")
    ax[1].set_ylabel("Curvature (m$^{-1}$)")
    # Wavelet
    # norm = MidPointNorm(midpoint=0)
    ax2_t = ax[2].twinx()
    im = ax2_t.pcolormesh(s_curvature, cwt_period, wave, cmap=cmap)
    im = ax[2].pcolormesh(s_curvature, scales, wave, cmap=cmap)

    ax[2].set_xlabel("Distance (m)")
    ax[2].set_ylabel("Scale (m)")
    ax2_t.set_ylabel("Wavelength (m)")
    ax[2].invert_yaxis()
    ax2_t.invert_yaxis()
    # ax[2].set_ylim([np.max(scales), np.min(scales)])
    # ax2_t.set_ylim([np.max(cwt_period), np.min(cwt_matrix)])
    if (zc_lines is not None) and (zc_sign is not None):
        for czc, z_c in enumerate(zc_lines):
            if isinstance(z_c, int):
                continue
            if zc_sign[czc] == 1:
                ax2_t.plot(
                    s_curvature[z_c[:, 1]], cwt_period[z_c[:, 0]], "-r", lw=1
                )
            else:
                ax2_t.plot(
                    s_curvature[z_c[:, 1]], cwt_period[z_c[:, 0]], "-b", lw=1
                )

    if poly is not None:
        for cpol, p in enumerate(poly):
            n_pts = p.shape[0] // 2
            ax2_t.plot(
                s_curvature[p[0:n_pts, 1]], cwt_period[p[0:n_pts, 0]], "--k"
            )

    if ml_tree is not None:
        s_c_peak = np.empty_like(peak_col) * np.nan
        s_c_peak[~np.isnan(peak_col)] = s_curvature[
            peak_col[~np.isnan(peak_col)].astype(int)
        ]
        scales_peak = np.empty_like(peak_row) * np.nan
        scales_peak[~np.isnan(peak_row)] = cwt_period[
            peak_row[~np.isnan(peak_row)].astype(int)
        ]

        WTFunc.plot_tree(ml_tree, s_c_peak, scales_peak, ax=ax2_t, **kwargs)
    if regions is not None:
        regions_2 = copy.deepcopy(regions)
        regions_2[:, 0] = cwt_period[regions[:, 0].astype(int)]
        regions_2[:, 1] = cwt_period[regions[:, 1].astype(int)]
        regions_2[:, 2] = s_curvature[regions[:, 2].astype(int)]
        regions_2[:, 3] = s_curvature[regions[:, 3].astype(int)]
        WTFunc.plot_regions(
            regions_2, ax=ax2_t, color="grey", linestyle="--", lw=0.5
        )

    ax[2].set_yscale("log")
    ax2_t.set_yscale("log")

    # Finish Figure
    fig.subplots_adjust(left=0.10, right=0.87)
    cbar_ax = fig.add_axes([0.91, 0.18, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Scaled Wavelet Transformed")
    if save:
        utl.cr_folder(path)
        plt.savefig(path + name + ".png", dpi=300)
        plt.close("all")
    else:
        plt.show()

    return fig, ax


def plot_river_with_plotly(
    river,
    tree: bool = False,
    meanders: bool = False,
    curvature_side: int = 1,
    so: int = 0,
    mapbox_token: Union[str, None] = None,
    projection: str = "esri:102003",
    data_source: str = "original",
) -> None:
    """plot the river transect with the meanders and the tree using plotly.

    Args:
        river (RiverTransect): River transect object.
        tree (bool, optional): flag to plot the tree. Defaults to False.
        meanders (bool, optional): flag to plot meanders. Defaults to False.
        curvature_side (int, optional): curveture side of meanders to plot.
            This parameter can be 1 or -1. Defaults to 1.
        so (int, optional): Stream order to plot. Only used if data_source is
            'smooth'. If 0 all stream orders will be plotted. Defaults to 0.
        mapbox_token (Union[str, None], optional): mapbox token. This will
            allow to plot the figure with the satellite image at the background.
            Defaults to None.
        projection (str, optional): projection of the current data. This is
            needed to convert the data to WGS84 for mapbox plotting.
            Defaults to "esri:102003".
        data_source (str, optional): data source use for plotting.
            Defaults to "original".
    """

    if mapbox_token is not None:
        satellite = True
        wgs84 = pyproj.CRS("EPSG:4326")
        nad1983 = pyproj.CRS(projection)
        transformer = pyproj.Transformer.from_crs(nad1983, wgs84)
    else:
        satellite = False

    # fig = make_subplots(rows=1, cols=1,
    #                     specs=[[{'type': 'mapbox'}], [{}]])
    if satellite:
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "mapbox"}]])
    else:
        fig = make_subplots(rows=1, cols=1)
    if data_source == "original":
        x = river.x
        y = river.y
    elif data_source == "smooth":
        x = river.x_smooth[so]
        y = river.y_smooth[so]

    if tree:
        ml_tree = copy.deepcopy(river.cwt_ml_tree[so])
        planimetry_coords = river.cwt_planimetry_coords[so]
        peak_pwr = river.cwt_peak_pwr[so]
        # peak_row = river.cwt_peak_row[so]
        # peak_col = river.cwt_peak_col[so]
        ml_tree = WTFunc.check_conn(ml_tree)
        nc = WTFunc.n_child(ml_tree)

    if satellite:
        y, x = transformer.transform(x, y)

    # c = river.c[so]
    # s_curvature = river.s_curvature[so]

    # Add River
    if satellite:
        fig.add_trace(
            go.Scattermapbox(
                lon=x,
                lat=y,
                mode="lines",
                name="River",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
        means_x = np.mean(x)
        means_y = np.mean(y)

        if tree:
            x_c = planimetry_coords[:, 0]
            y_c = planimetry_coords[:, 1]
            # print(x_c.shape, y_c.shape, ml_tree.shape, nc.shape)
            # print(peak_pwr.shape, peak_row.shape, peak_col.shape)
            for cp, c_n in enumerate(ml_tree):
                if c_n == -1:
                    continue
                else:
                    xp = [x_c[cp]]
                    yp = [y_c[cp]]
                    x_con = [x_c[cp], x_c[c_n]]
                    y_con = [y_c[cp], y_c[c_n]]
                    fig.add_trace(
                        go.Scatter(
                            x=x_con,
                            y=y_con,
                            mode="lines",
                            name="Connection",
                            line=dict(color="green", width=2),
                        ),
                        row=1,
                        col=1,
                    )

            # Plot points over
            for cp, c_n in enumerate(ml_tree):
                if c_n == -1:
                    if nc[cp] == 0:
                        xp = [x_c[cp]]
                        yp = [y_c[cp]]
                        # plot root node
                        fig.add_trace(
                            go.Scatter(
                                x=xp,
                                y=yp,
                                mode="lines",
                                name="Root Node",
                                marker=dict(color="cyan"),
                                hovertemplate=(
                                    "id: "
                                    + str(cp)
                                    + "<br>pwr: "
                                    + str(peak_pwr[cp])
                                ),
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        continue
                else:
                    xp = [x_c[cp]]
                    yp = [y_c[cp]]
                    fig.add_trace(
                        go.Scatter(
                            x=xp,
                            y=yp,
                            mode="lines",
                            name="Node",
                            marker=dict(color="cyan"),
                            hovertemplate=(
                                "id: "
                                + str(cp)
                                + "<br>pwr: "
                                + str(peak_pwr[cp])
                            ),
                        ),
                        row=1,
                        col=1,
                    )

        if meanders:
            # Adding Lines
            for id_m in river.id_meanders:
                meander = river.meanders[id_m]
                if meander.curvature_side == curvature_side:
                    x_m = meander.x
                    y_m = meander.y
                    y_m, x_m = transformer.transform(x_m, y_m)
                    fig.add_trace(
                        go.Scattermapbox(
                            lon=x_m,
                            lat=y_m,
                            mode="lines",
                            name=f"{id_m}",
                            line=dict(color="red", width=2),
                            hovertemplate=(
                                "id: "
                                + str(id_m)
                                + "<br>curvature: "
                                + str(curvature_side)
                                + f'<br>lambda: {meander.data["lambda"]:0.3f}'
                                + f'<br>l: {meander.data["l"]:0.3f}'
                                + f'<br>sn: {meander.data["sinuosity"]:0.3f}'
                                + f'<br>sk: {meander.data["skewness"]:0.3f}'
                                + f'<br>fl: {meander.data["flatness"]:0.3f}'
                            ),
                        ),
                        row=1,
                        col=1,
                    )
            # Adding Lines
            for id_m in river.id_meanders:
                meander = river.meanders[id_m]
                if meander.curvature_side == curvature_side:
                    x_m = meander.x
                    y_m = meander.y
                    y_m, x_m = transformer.transform(x_m, y_m)
                    # Starting Point
                    fig.add_trace(
                        go.Scattermapbox(
                            lon=np.array([x_m[0]]),
                            lat=np.array([y_m[0]]),
                            mode="markers",
                            name=f"{id_m}_start",
                            marker=dict(color="green"),
                            hovertemplate=(
                                "id: "
                                + str(id_m)
                                + " <br>curvature: "
                                + str(curvature_side)
                                + f'<br>lambda: {meander.data["lambda"]:0.3f}'
                                + f'<br>l: {meander.data["l"]:0.3f}'
                                + f'<br>sn: {meander.data["sinuosity"]:0.3f}'
                                + f'<br>sk: {meander.data["skewness"]:0.3f}'
                                + f'<br>fl: {meander.data["flatness"]:0.3f}'
                            ),
                        ),
                        row=1,
                        col=1,
                    )
                    # Ending Point
                    fig.add_trace(
                        go.Scattermapbox(
                            lon=np.array([x_m[-1]]),
                            lat=np.array([y_m[-1]]),
                            mode="markers",
                            name=f"{id_m}_end",
                            marker=dict(color="cyan"),
                            hovertemplate=(
                                "id: "
                                + str(id_m)
                                + "<br>curvature: "
                                + str(curvature_side)
                                + f'<br>lambda: {meander.data["lambda"]:0.3f}'
                                + f'<br>l: {meander.data["l"]:0.3f}'
                                + f'<br>sn: {meander.data["sinuosity"]:0.3f}'
                                + f'<br>sk: {meander.data["skewness"]:0.3f}'
                                + f'<br>fl: {meander.data["flatness"]:0.3f}'
                            ),
                        ),
                        row=1,
                        col=1,
                    )

        fig.update_layout(
            mapbox=dict(
                accesstoken=mapbox_token,
                bearing=0,
                center=dict(lat=np.mean(means_y), lon=np.mean(means_x)),
                pitch=0,
                zoom=8,
                style="satellite",
            )
        )

    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="River",
                line=dict(color="blue", width=2),
            ),
            row=1,
            col=1,
        )
        if tree:
            x_c = planimetry_coords[:, 0]
            y_c = planimetry_coords[:, 1]
            # print(x_c.shape, y_c.shape, ml_tree.shape, nc.shape)
            # print(peak_pwr.shape, peak_row.shape, peak_col.shape)
            for cp, c_n in enumerate(ml_tree):
                if c_n == -1:
                    continue
                else:
                    xp = [x_c[cp]]
                    yp = [y_c[cp]]
                    x_con = [x_c[cp], x_c[c_n]]
                    y_con = [y_c[cp], y_c[c_n]]
                    fig.add_trace(
                        go.Scatter(
                            x=x_con,
                            y=y_con,
                            mode="lines",
                            name="Connection",
                            line=dict(color="green", width=2),
                        ),
                        row=1,
                        col=1,
                    )

            # Plot points over
            for cp, c_n in enumerate(ml_tree):
                if c_n == -1:
                    if nc[cp] == 0:
                        xp = [x_c[cp]]
                        yp = [y_c[cp]]
                        # plot root node
                        fig.add_trace(
                            go.Scatter(
                                x=xp,
                                y=yp,
                                mode="lines",
                                name="Root Node",
                                marker=dict(color="cyan"),
                                hovertemplate=(
                                    "id: "
                                    + str(cp)
                                    + "<br>pwr: "
                                    + str(peak_pwr[cp])
                                ),
                            ),
                            row=1,
                            col=1,
                        )
                    else:
                        continue
                else:
                    xp = [x_c[cp]]
                    yp = [y_c[cp]]
                    fig.add_trace(
                        go.Scatter(
                            x=xp,
                            y=yp,
                            mode="lines",
                            name="Node",
                            marker=dict(color="cyan"),
                            hovertemplate=(
                                "id: "
                                + str(cp)
                                + "<br>pwr: "
                                + str(peak_pwr[cp])
                            ),
                        ),
                        row=1,
                        col=1,
                    )
        fig.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    # Add Curvature
    # fig.add_trace(
    #     go.Scatter(
    #     x=s_curvature, y=c, mode='lines',
    #     name=f'River',
    #     line=dict(color='black', width=2)),
    #     row=2, col=1
    # )

    # Update layout
    fig.update_layout(
        title=f"River: {river.id_value}",
        xaxis=dict(title="x"),
        yaxis=dict(title="y"),
        showlegend=False,
        hovermode="closest",
        autosize=True,
    )
    fig.show()
    return


def plot_rivers_plotly(
    rivers,
    comids,
    data_source="resample",
    mapbox_token=None,
    current_crs="epsg:4326",
    zoom=5,
):

    traces = []
    # Extracting Info
    x_all = []
    y_all = []
    means_x = []
    means_y = []
    for st_river in comids:
        rivers[st_river].data_source = data_source
        (
            x,
            y,
            s,
        ) = rivers[st_river]._extract_data_source()
        x_all.append(x)
        y_all.append(y)
        means_x.append(np.mean(x_all[-1]))
        means_y.append(np.mean(y_all[-1]))

    if mapbox_token is not None:
        if current_crs != "epsg:4326":
            wgs84 = pyproj.CRS("EPSG:4326")
            projected = pyproj.CRS(current_crs)
            transformer = pyproj.Transformer.from_crs(projected, wgs84)
        else:
            transformer = None
        # Transforming coordinates
        for i, st_river in enumerate(comids):
            if transformer is not None:
                y_all[i], x_all[i] = transformer.transform(x_all[i], y_all[i])
            traces.append(
                go.Scattermapbox(
                    lat=np.array(y_all[i]),
                    lon=np.array(x_all[i]),
                    mode="lines",
                    name=int(st_river),
                )
            )
            means_y[i], means_x[i] = transformer.transform(
                means_x[i], means_y[i]
            )

        # set the layout
        layout = go.Layout(
            title="Rivers",
            autosize=True,
            hovermode="closest",
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
            mapbox=dict(
                accesstoken=mapbox_token,
                bearing=0,
                center=dict(lat=np.mean(means_y), lon=np.mean(means_x)),
                pitch=0,
                zoom=zoom,
                style="satellite",
            ),
        )
        fig = go.Figure(data=traces, layout=layout)
    else:
        for st_river in comids:
            rivers[st_river].data_source = data_source
            (
                x,
                y,
                s,
            ) = rivers[st_river]._extract_data_source()
            # x = rivers[st_river].x
            # y = rivers[st_river].y
            traces.append(
                go.Scatter(x=x, y=y, mode="lines", name=int(st_river))
            )

        # set the layout
        layout = go.Layout(
            title="Rivers",
            xaxis=dict(title="x"),
            yaxis=dict(title="y"),
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
        )
        fig = go.Figure(data=traces, layout=layout)
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def plot_rivers_matplotlib(
    rivers,
    comids: Union[str, list, np.ndarray],
    data_source: str = "resample",
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """plot the rivers using matplotlib.

    Args:
        rivers (RiverDatasets): RiverDatasets object.
        comids (Union[str, list, np.ndarray]): list of comids to be plotted.
        data_source (str, optional): data source of the coordinates.
            The options are 'original' and 'resample'. Defaults to "resample".
        **kwargs: additional arguments to be passed to the ax.plot.

    Returns:
        Tuple[plt.Figure, plt.Axes]: figure and axes of the plot.
    """
    if isinstance(comids, str):
        comids = [comids]

    fig, ax = plt.subplots()
    # Extracting Info
    x_all = []
    y_all = []
    for st_river in comids:
        rivers[st_river].data_source = data_source
        (
            x,
            y,
            s,
        ) = rivers[st_river]._extract_data_source()
        x_all.append(x)
        y_all.append(y)

    # Transforming coordinates
    for i, st_river in enumerate(comids):
        ax.plot(x_all[i], y_all[i], label=st_river, **kwargs)

    # set the layout
    ax.set_title("Rivers")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.legend()
    return fig, ax


def plot_tree_from_anytree(
    x: np.ndarray,
    y: np.ndarray,
    s_curvature: np.ndarray,
    wavelength: np.ndarray,
    wave: np.ndarray,
    tree_scales: dict,
    gws: np.ndarray,
    peaks_gws: np.ndarray,
    id_river: Union[int, float, str],
    coi: Union[np.ndarray, None] = None,
    tree_ids: Union[list, None] = None,
    node_ids: Union[int, None] = None,
    min_s: Union[np.ndarray, None] = None,
    include_removed: bool = False,
    scale_by_width: bool = False,
    title: Union[str, None] = None,
    **kwargs,
) -> Tuple[plt.Figure, plt.Axes]:
    """plot the river, the curvature, and the wavelet response using anytree
    tree.

    Args:
        x (np.ndarray): x coordinates of the river.
        y (np.ndarray): y coordinates of the river.
        s_curvature (np.ndarray): Arc length of the river.
        wavelength (np.ndarray): Wavelength of the wavelet response.
        wave (np.ndarray): power of the wavelet response.
        tree_scales (dict): treescales object.
        gws (np.ndarray): generalized wavelet spectrum.
        peaks_gws (np.ndarray): peaks of the generalized wavelet spectrum.
        id_river (Union[int, float, str]): river id.
        coi (Union[np.ndarray, None], optional): cone of influence.
            Defaults to None.
        tree_ids (Union[list, None], optional): tree ids to plot.
            Defaults to None.
        node_ids (Union[int, None], optional): node ids to plot.
            Defaults to None.
        min_s (Union[np.ndarray, None], optional): cut in this spectrum.
            Defaults to None.
        include_removed (bool, optional): include removed branches.
            Defaults to False.
        scale_by_width (bool, optional): scale distance by width.
            Defaults to False.
        title (Union[str, None], optional): title of the figure.
            Defaults to None.

    Raises:
        ValueError: node_ids must be None if more than one tree_id

    Returns:
        Tuple[plt.Figure, plt.Axes]: figure and Axes of the plot.
    """

    if tree_ids is not None:
        all_trees = False
        if isinstance(tree_ids, int) or isinstance(tree_ids, np.int64):
            tree_ids = [tree_ids]
    else:
        tree_ids = list(tree_scales.trees)
        all_trees = True
    if node_ids is not None and len(tree_ids) == 1:
        if isinstance(node_ids, int) or isinstance(node_ids, np.int64):
            node_ids = [node_ids]
    elif node_ids is not None and len(tree_ids) > 1:
        raise ValueError("node_ids must be None if more than one tree_id")

    # ========================
    # Plotting
    # ========================
    fig, ax = plt.subplots(2, 1, figsize=(8, 5), sharey="row")
    # add title
    # fig.suptitle(f'{id_river}')
    # -----------------
    # Planimetry
    # -----------------
    if title is not None:
        ax[0].set_title(f"(a) {title}")
    else:
        ax[0].set_title(f"(a) {id_river}")
    ax[0].plot(x, y, "-k", label="River centerline")
    ax[0].set_aspect("equal")
    ax[0].axis("off")
    # -----------------
    # Spectral
    # -----------------
    im = ax[1].pcolormesh(s_curvature, wavelength, wave, cmap="Spectral")
    if scale_by_width:
        ax[1].set_xlabel("$s^*$")
    else:
        ax[1].set_xlabel("$s$ (m)")
    # ax[1, 1].set_ylabel('Wavelength [m]')
    ax[1].set_ylim([wavelength[-1], wavelength[0]])
    # if gamma_width is not None:
    #     ax[1].plot(s_curvature, gamma_width, '-k', linewidth=2)
    if coi is not None:
        ax[1].fill_between(
            s_curvature,
            coi * 0 + wavelength[-1],
            coi,
            facecolor="none",
            edgecolor="#00000040",
            hatch="x",
        )
    # ax[1].invert_yaxis()
    ax[1].set_yscale("log")
    ax[1].set_yticklabels([])
    ax[1].set_title("(b) CWT of Curvature with Scale Space Tree")

    # Plot tree
    for tree_id in tree_ids:
        root_node = tree_scales[tree_id]
        if node_ids is None:
            nodes = [root_node] + list(root_node.descendants)
        else:
            nodes = node_ids
        for node in nodes:
            removed_meander = node.is_leaf and not (node.is_meander)
            if not include_removed and removed_meander:
                continue
            # Plot in planimetry
            ax[0] = plot_node(ax[0], node, x_var="x_c", y_var="y_c", **kwargs)
            ax[1] = plot_node(
                ax[1], node, x_var="s_c", y_var="wavelength_c", **kwargs
            )
            if node.parent is not None:
                ax[0].plot(
                    [node.x_c, node.parent.x_c],
                    [node.y_c, node.parent.y_c],
                    color="r",
                    **kwargs,
                )
                ax[1].plot(
                    [node.s_c, node.parent.s_c],
                    [node.wavelength_c, node.parent.wavelength_c],
                    color="r",
                    **kwargs,
                )

    # Finish Plot
    fig.subplots_adjust(left=0.20, right=0.90, bottom=0.15)
    # cbar_ax = fig.add_axes([0.91, 0.18, 0.01, 0.7])
    # cbar = fig.colorbar(im, cax=cbar_ax)
    # cbar.set_label('Scaled Wavelet Transformed')
    pos = ax[1].get_position()
    cbar_ax = fig.add_axes([0.91, pos.y0, 0.01, pos.height])
    cbar = fig.colorbar(im, cax=cbar_ax)
    # Add Label
    if scale_by_width:
        cbar.ax.set_ylabel(
            r"$\log_2(|W^*_{n,c}|^2)$", rotation=270, labelpad=15
        )
    else:
        cbar.ax.set_ylabel(
            r"$\log_2(|W_{n,c}|^2$ (m$^{-2}))$", rotation=270, labelpad=15
        )
    # Peaks_min
    if min_s is not None:
        for s_m in min_s:
            ax[1].axvline(s_m, color="r", linestyle="--")
            i_min = np.argmin(np.abs(s_curvature - s_m))
            ax[0].axvline(x[i_min], color="r", linestyle="--")
    # Add GWS
    if len(tree_ids) == 1 and not (all_trees):
        gws = root_node.gws
        peaks_gws = root_node.gws_peak_wavelength
        idx_st = root_node.idx_leaf_start
        idx_end = root_node.idx_leaf_end
        x_clip = x[idx_st : idx_end + 1]
        y_clip = y[idx_st : idx_end + 1]
        ax[0].plot(x_clip, y_clip, "-b", linewidth=1)
        w_st = s_curvature[idx_st]
        w_end = s_curvature[idx_end]
        ax[1].axvline(w_st, color="k", linestyle="--")
        ax[1].axvline(w_end, color="k", linestyle="--")
    ax_gws = fig.add_axes([0.1, pos.y0, 0.09, pos.height])
    ax_gws.plot(gws, wavelength, "-k")
    for peak in peaks_gws:
        ax_gws.axhline(peak, color="b", linestyle="--")
        ax[1].axhline(peak, color="b", linestyle="--")
    if scale_by_width:
        ax_gws.set_xlabel(r"$|\overline{W^*_{n,c,GWS}}|^2$")
        ax_gws.set_ylabel(r"$\lambda^*$")
    else:
        ax_gws.set_xlabel("$|\\overline{W_{n,c,GWS}}|^2$\n(m$^{-2}$)")
        ax_gws.set_ylabel(r"$\lambda$ (m)")
    ax_gws.set_yscale("log")
    ax_gws.set_xscale("log")
    ax_gws.invert_xaxis()
    ax_gws.set_ylim([wavelength[-1], wavelength[0]])
    ax_gws.set_title("GWS")

    return fig, ax


def plot_node(
    ax: plt.Axes, node: Node, x_var: str, y_var: str, **kwargs
) -> plt.Axes:
    """plot node in the planimetry.

    Args:
        ax (plt.Axes): axes of the plot.
        node (Node): Node object.
        x_var (str): variable name of the x coordinate.
        y_var (str): variable name of the y coordinate.

    Returns:
        plt.Axes: Axes of the plot.
    """

    x = node.__dict__[x_var]
    y = node.__dict__[y_var]

    # Check type of the node
    if node.is_meander == 1:
        ax.plot(x, y, "o", color="b", **kwargs, label="Meander")
    elif node.is_leaf:
        ax.plot(x, y, "o", color="k", **kwargs, label="Removed Node")
    elif node.root_node == 1:
        ax.plot(x, y, "o", color="g", **kwargs, label="Root Node")
    else:
        ax.plot(x, y, "o", color="r", **kwargs, label="Parent Node")

    return ax


def plot_meander_matplotlib(
    x_river: np.ndarray,
    y_river: np.ndarray,
    x_meander: np.ndarray,
    y_meander: np.ndarray,
) -> plt.Figure:
    """plot meander in the river.

    Args:
        x_river (np.ndarray): x coordinates of the river.
        y_river (np.ndarray): y coordinates of the river.
        x_meander (np.ndarray): x coordinates of the meander.
        y_meander (np.ndarray): y coordinates of the meander.
        **kwargs: additional arguments to be passed to the ax.plot.

    Returns:
        plt.Figure: Figure of the plot.
    """
    f = plt.figure(figsize=(5, 5))
    plt.plot(x_river, y_river, "-k")
    plt.plot(x_meander, y_meander, "-r")
    plt.axis("equal")
    # Zoom to meander
    x_coords = np.vstack([x_meander, y_meander]).T
    distance_meander = RF.get_reach_distances(x_coords)[1]
    plt.xlim(
        [
            np.min(x_meander) - distance_meander * 2,
            np.max(x_meander) + distance_meander * 2,
        ]
    )
    plt.ylim(
        [
            np.min(y_meander) - distance_meander * 2,
            np.max(y_meander) + distance_meander * 2,
        ]
    )
    return f


def plot_river_spectrum_compiled(
    river,
    only_significant: bool = True,
    idx_data: Union[np.ndarray, None] = None,
) -> Union[plt.Figure, plt.Axes]:
    """plot river spectrum.

    Args:
        river (RiverTransect): RiverTransect object.
        only_significant (bool, optional): flag to show only significant
            meanders. Defaults to True.
        idx_data (Union[np.ndarray, None], optional): index data to plot.
            Defaults to None.

    Returns:
        Union[plt.Figure, plt.Axes]: Figure and Axes of the plot.
    """
    fs = 10
    mpl.rcParams["font.size"] = fs
    # Extract Information
    cmap = "YlGnBu"
    # id_river = river.id_value
    x = river.x
    y = river.y
    s = river.s
    angle = river.angle
    c = river.c
    wavelen_c = river.cwt_wavelength_c
    wavelen_angle = river.cwt_wavelength_angle
    power_c = copy.deepcopy(river.cwt_power_c)
    coi_c = river.cwt_coi_c
    power_angle = copy.deepcopy(river.cwt_power_angle)
    coi_angle = river.cwt_coi_angle
    gws_c = river.cwt_gws_c
    gws_angle = river.cwt_gws_angle
    sawp_c = river.cwt_sawp_c
    sawp_angle = river.cwt_sawp_angle
    scale_by_width = river.scale_by_width

    labels = [
        r"$C$ (m$^{-1}$)",
        r"$\lambda$ (m)",
        r"$\theta$",
        r"$s$ (m)",
        r"$|\overline{W_{n,c,GWS}}|^2$ (m$^{-2}$)",
        "$|\\overline{W_{n,c,SAWP}}|^2$\n(m$^{-2}$)",
        "$|\\overline{W_{n,\\theta,GWS}}|^2$\n(deg)",
        "$|\\overline{W_{n,\\theta,SAWP}}|^2$\n(deg)",
    ]
    if scale_by_width:
        labels = [
            r"$C^*$",
            r"$\lambda^*$",
            r"$\theta$",
            r"$s^*$",
            r"$|\overline{W^*_{n,c,GWS}}|^2$",
            r"$|\overline{W^*_{n,c,SAWP}}|^2$",
            r"$|\overline{W_{n,\theta,GWS}}|^2$",
            r"$|\overline{W_{n,\theta,SAWP}}|^2$",
        ]

    if only_significant:
        power_c = river.cwt_power_c_sig[:]
        gws_c = river.cwt_gws_c_sig[:]
        sawp_c = river.cwt_sawp_c_sig[:]
        power_angle = river.cwt_power_angle_sig[:]
        gws_angle = river.cwt_gws_angle_sig[:]
        sawp_angle = river.cwt_sawp_angle_sig[:]
        power_c[power_c == 0] = np.nan
        power_angle[power_angle == 0] = np.nan

    fig, ax = plt.subplots(7, 1, figsize=(8, 10))

    ax = ax.ravel()
    # ------------------------
    # Planimetry Coordinate
    # ------------------------
    i_d = 0
    ax[i_d].plot(x, y, "k", linewidth=1.5)
    if idx_data is not None:
        ax[i_d].plot(x[idx_data], y[idx_data], "ro")
    ax[i_d].set_aspect("equal")
    # Extend y axis
    ax[i_d].set_ylim([np.min(y) - np.std(y), np.max(y) + np.std(y)])
    ax[i_d].set_xticks([])
    ax[i_d].set_yticks([])
    # Remove axis lines
    ax[i_d].spines["top"].set_visible(False)
    ax[i_d].spines["right"].set_visible(False)
    ax[i_d].spines["bottom"].set_visible(False)
    ax[i_d].spines["left"].set_visible(False)
    ax[i_d].set_title("(a) River Planimetry")

    # =============
    # Curvature
    # =============
    i_d = 1
    ax[i_d].plot(s, c, "k", linewidth=0.5)
    if idx_data is not None:
        for idx in idx_data:
            ax[i_d].axvline(s[idx], color="r", linestyle="--")
    ax[i_d].set_xlim([np.min(s), np.max(s)])
    ax[i_d].set_ylabel(labels[0])
    ax[i_d].set_title("(b) Curvature")

    # ------------------------
    # Power Curvature
    # ------------------------
    i_d = 2
    ax[i_d].pcolormesh(s, wavelen_c, power_c, shading="auto", cmap=cmap)
    if idx_data is not None:
        for idx in idx_data:
            ax[i_d].axvline(s[idx], color="r", linestyle="--")
    # Plot COI
    ax[i_d].fill_between(
        s,
        coi_c * 0 + wavelen_c[-1],
        coi_c,
        facecolor="none",
        edgecolor="#00000040",
        hatch="x",
    )
    ax[i_d].set_yscale("log")
    ax[i_d].set_ylim([np.max(wavelen_c), np.min(wavelen_c)])
    ax[i_d].set_xlim([np.min(s), np.max(s)])
    ax[i_d].set_ylabel(labels[1])
    # ax[i_d].set_yticklabels([])
    # for peak in gws_peak_wavelen_c:
    #     ax[i_d].axhline(peak, color='r', linestyle='--')

    # ------------------------
    # SAWP
    # ------------------------
    i_d = 3
    ax[i_d].plot(s, sawp_c, "k", linewidth=0.5)
    ax[i_d].text(
        0.46,
        0.8,
        "SAWP",
        transform=ax[i_d].transAxes,
    )
    ax[i_d].set_xlim([np.min(s), np.max(s)])
    ax[i_d].set_ylim(top=np.max(sawp_c) * 1.5)
    ax[i_d].set_xlabel(labels[3])
    ax[i_d].set_ylabel(labels[5])

    # =============
    # Angle
    # =============
    i_d = 4
    ax[i_d].plot(s, angle, "k", linewidth=0.5)
    if idx_data is not None:
        for idx in idx_data:
            ax[i_d].axvline(s[idx], color="r", linestyle="--")
    ax[i_d].set_xlim([np.min(s), np.max(s)])
    ax[i_d].set_ylabel(labels[2])
    ax[i_d].set_title("(c) Direction Angle")

    # ------------------------
    # Power Angle
    # ------------------------
    i_d = 5
    ax[i_d].pcolormesh(s, wavelen_angle, power_angle, shading="auto", cmap=cmap)
    if idx_data is not None:
        for idx in idx_data:
            ax[i_d].axvline(s[idx], color="r", linestyle="--")
    ax[i_d].fill_between(
        s,
        coi_angle * 0 + wavelen_angle[-1],
        coi_angle,
        facecolor="none",
        edgecolor="#00000040",
        hatch="x",
    )
    ax[i_d].set_yscale("log")
    ax[i_d].set_ylim([np.max(wavelen_angle), np.min(wavelen_angle)])
    ax[i_d].set_xlim([np.min(s), np.max(s)])
    ax[i_d].set_ylabel(labels[1])
    # ax[i_d].set_yticklabels([])
    # for peak in gws_peak_wavelen_angle:
    #     ax[i_d].axhline(peak, color='r', linestyle='--')

    # ------------------------
    # SAWP
    # ------------------------
    i_d = 6
    ax[i_d].plot(s, sawp_angle, "k", linewidth=0.5)
    ax[i_d].text(
        0.46,
        0.8,
        "SAWP",
        transform=ax[i_d].transAxes,
    )
    ax[i_d].set_xlim([np.min(s), np.max(s)])
    ax[i_d].set_ylim(top=np.max(sawp_angle) * 1.5)
    ax[i_d].set_ylabel(labels[7])
    ax[i_d].set_xlabel(labels[3])

    plt.tight_layout()
    # fig.subplots_adjust(left=0.25, right=0.90)
    fig.subplots_adjust(left=0.12, right=0.80)

    # ========================
    # Move figures
    # ========================
    # Curvature
    ax[1].set_xticklabels([])
    pos = ax[2].get_position()
    ax[2].set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height + 0.11])
    ax[2].set_xticklabels([])
    pos = ax[3].get_position()
    ax[3].set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])

    # Direction Angle
    ax[4].set_xticklabels([])
    pos = ax[5].get_position()
    ax[5].set_position([pos.x0, pos.y0 - 0.04, pos.width, pos.height + 0.11])
    ax[5].set_xticklabels([])
    pos = ax[6].get_position()
    ax[6].set_position([pos.x0, pos.y0 + 0.03, pos.width, pos.height])

    # ========================
    # Add colorbars
    # ========================
    # Add Colorbars
    # pos = ax[2].get_position()
    # ax_c = fig.add_axes([0.91, pos.y0, 0.02, pos.height])
    # cbar_c = fig.colorbar(im_c, cax=ax_c)
    # pos = ax[3].get_position()
    # ax_c = fig.add_axes([pos.x0, pos.y0-0.05, pos.width, 0.02])
    # cbar_c = fig.colorbar(im_c, cax=ax_c, orientation='horizontal')

    # pos = ax[5].get_position()
    # ax_angle = fig.add_axes([0.91, pos.y0, 0.02, pos.height])
    # cbar_angle = fig.colorbar(im_angle, cax=ax_angle)
    # pos = ax[6].get_position()
    # ax_angle = fig.add_axes([pos.x0, pos.y0-0.05, pos.width, 0.02])
    # cbar_angle = fig.colorbar(im_angle, cax=ax_angle, orientation='horizontal')

    # ========================
    # Add GWS
    # ========================
    # Add GWS c
    pos = ax[2].get_position()
    # ax_gws_c = fig.add_axes([0.1, pos.y0, 0.135, pos.height])
    ax_gws_c = fig.add_axes([0.815, pos.y0, 0.135, pos.height])
    ax_gws_c.plot(gws_c, wavelen_c, "k", linewidth=0.5)
    ax_gws_c.set_yscale("log")
    ax_gws_c.set_ylim([np.max(wavelen_c), np.min(wavelen_c)])
    # ax_gws_c.set_ylabel(labels[1])
    ax_gws_c.set_xlabel(labels[4])
    ax_gws_c.set_yticklabels([])
    ax_gws_c.set_title("GWS")
    # for peak in gws_peak_wavelen_c:
    #     ax_gws_c.axhline(peak, color='r', linestyle='--')

    pos = ax[5].get_position()
    # ax_gws_angle = fig.add_axes([0.1, pos.y0, 0.135, pos.height])
    ax_gws_angle = fig.add_axes([0.815, pos.y0, 0.135, pos.height])
    ax_gws_angle.plot(gws_angle, wavelen_angle, "k", linewidth=0.5)
    ax_gws_angle.set_yscale("log")
    ax_gws_angle.set_ylim([np.max(wavelen_angle), np.min(wavelen_angle)])
    # ax_gws_angle.set_ylabel(labels[1])
    ax_gws_angle.set_xlabel(labels[6])
    ax_gws_angle.set_title("GWS")
    ax_gws_angle.set_yticklabels([])
    # for peak in gws_peak_wavelen_angle:
    #     ax_gws_angle.axhline(peak, color='r', linestyle='--')
    # plt.tight_layout()

    # Literals
    # fig.text(0.1, 0.9, '(A)', weight='bold', fontsize=fs + 2)

    return fig, ax
