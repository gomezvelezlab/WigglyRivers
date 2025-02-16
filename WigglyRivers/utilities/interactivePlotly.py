# -*- coding: utf-8 -*-
# ______________________________________________________________________________
# ______________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 2025-02-13
# ______________________________________________________________________________
# ______________________________________________________________________________
"""
The functions here are used to have the ineractive plots for the Meander
Characterization App.
"""
# ------------------------
# Importing Modules
# ------------------------
import copy
from typing import Union

# Data Managment
import numpy as np

# Graphs
import pyproj
import plotly.graph_objects as go

# Personal libraries


# ------------------------
# Functions
# ------------------------
def plot_interactive_river_plain(
    x: Union[list, np.ndarray],
    y: Union[list, np.ndarray],
    clicked_points: list = [],
    meander_ids: list = [],
    river_obj=None,
    inflection_flag: bool = False,
    mapbox_token: Union[bool, None] = None,
    current_crs: str = "epsg:4326",
    zoom: int = 5,
) -> go.FigureWidget:
    """This function plots an interactive plot to detect meanders in a river
    planform. This function can only be used in a Jupyter Notebook.

    Args:
        x (Union[list, np.ndarray]): x coordinates of the river.
        y (Union[list, np.ndarray]): y coordinates of the river.
        clicked_points (list, optional): list of clicked points. Defaults to [].
        meander_ids (list, optional): meander ids for the clicked points.
            Defaults to [].
        river_obj (Union[RiverTransect, None], optional): RiverTransect object.
            Defaults to None.
        inflection_flag (bool, optional): flag to plot infection points.
            Defaults to False.
        mapbox_token (Union[bool, None], optional): mapbox token. This is needed
            to have the satellite image in the background. If None the plot will
            have a white background. Defaults to None.
        current_crs (str, optional): Current projection of the data. This is
            needed to reproject the data a WGS84 for mapbox plotting.
            Defaults to "epsg:4326".
        zoom (int, optional): Zoom of the plot. Defaults to 5.

    Returns:
        go.FigureWidget: Figure widget of the interactive plot.
    """
    # Check if mapbox token is not None
    if mapbox_token is not None:
        satellite = True
        if current_crs != "epsg:4326":
            wgs84 = pyproj.CRS("EPSG:4326")
            projected = pyproj.CRS(current_crs)
            transformer = pyproj.Transformer.from_crs(projected, wgs84)
            y_lat, x_lon = transformer.transform(x, y)
        else:
            x_lon = copy.deepcopy(x)
            y_lat = copy.deepcopy(y)
        mean_x_lon = np.mean(x_lon)
        mean_y_lat = np.mean(y_lat)
    else:
        satellite = False

    # global meander_id
    traces = []
    if satellite:
        traces.append(
            go.Scattermapbox(
                lat=np.array(y_lat),
                lon=np.array(x_lon),
                mode="lines",
                name="River",
                line=dict(color="blue", width=2),
            )
        )
    else:
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name="River",
                line=dict(color="blue", width=2),
            )
        )
    f = go.FigureWidget(traces)
    # f.update_layout(showlegend=False)
    scatter = f.data[0]
    f.layout.hovermode = "closest"
    if satellite:
        f.update_layout(
            autosize=True,
            showlegend=True,
            mapbox=dict(
                accesstoken=mapbox_token,
                bearing=0,
                center=dict(lat=mean_y_lat, lon=mean_x_lon),
                pitch=0,
                zoom=zoom,
                style="satellite",
            ),
            width=800,
            height=500,
            margin=dict(l=20, r=20, t=20, b=20),
        )
    else:
        f.update_yaxes(scaleanchor="x", scaleratio=1)
        f.update_layout(
            plot_bgcolor="white",
            showlegend=True,
            autosize=True,
            width=800,
            height=500,
            margin=dict(l=0, r=0, t=0, b=0),
        )
        f.update_xaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            showgrid=False,
            zeroline=False,
        )
        f.update_yaxes(
            mirror=True,
            ticks="outside",
            showline=True,
            linecolor="black",
            showgrid=False,
            zeroline=False,
        )

    # Load Previous Meanders
    if river_obj is not None:
        f = load_meanders(
            f, river_obj, mapbox_token=mapbox_token, current_crs=current_crs
        )

    def update_point(trace, points, selector):
        meander_id = check_meander_id(meander_ids)
        for i in points.point_inds:
            # Save points
            clicked_points.append(i)
            hover_template = create_hover_template_meander(meander_id, None)

            if satellite:
                f.add_trace(
                    go.Scattermapbox(
                        lat=np.array(y_lat[i]),
                        lon=np.array(x_lon[i]),
                        mode="markers",
                        name=f"Meander {meander_id}",
                        marker=dict(color="red", size=10),
                        showlegend=False,
                        hovertemplate=hover_template,
                    )
                )
            else:
                f.add_trace(
                    go.Scatter(
                        x=[trace.x[i]],
                        y=[trace.y[i]],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name=f"Meander {meander_id}",
                        showlegend=False,
                        hovertemplate=hover_template,
                    )
                )

            if len(clicked_points) % 2 == 0:
                # Add lines between points
                idx_1 = clicked_points[-2]
                idx_2 = clicked_points[-1]
                idx_start = min(idx_1, idx_2)
                idx_end = max(idx_1, idx_2)
                # Add meander to river object
                if river_obj is not None:
                    river_obj.add_meander(
                        meander_id, ind_start=idx_start, ind_end=idx_end
                    )
                hover_template = create_hover_template_meander(
                    meander_id, river_obj
                )

                if satellite:
                    x_line = x_lon[idx_start : idx_end + 1]
                    y_line = y_lat[idx_start : idx_end + 1]
                    f.add_trace(
                        go.Scattermapbox(
                            lat=np.array(y_line),
                            lon=np.array(x_line),
                            mode="lines",
                            name=f"Meander {meander_id}",
                            line=dict(color="red", width=2),
                            hovertemplate=hover_template,
                        )
                    )
                else:
                    x_line = x[idx_start : idx_end + 1]
                    y_line = y[idx_start : idx_end + 1]
                    f.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="red", width=2),
                            name=f"Meander {meander_id}",
                            hovertemplate=hover_template,
                        )
                    )

                f.update_traces(
                    hovertemplate=hover_template,
                    selector=dict(name=f"Meander {meander_id}"),
                )

                # Append meander id
                meander_ids.append(meander_id)
                # Reset points
                # clicked_points = []

    scatter.on_click(update_point)
    return f


def check_meander_id(meander_ids: list) -> int:
    """Check meander ids.

    Args:
        meander_ids (list): Meander IDs.

    Returns:
        int: new meander id.
    """
    meander_id = 0
    if len(meander_ids) != 0:
        while meander_id in meander_ids:
            meander_id += 1

    return meander_id


def load_meanders(
    f: go.FigureWidget,
    river_obj,
    mapbox_token: str = None,
    current_crs: str = "epsg:4326",
) -> go.FigureWidget:
    """load current selection of meanders.

    Args:
        f (go.FigureWidget): plotly figure widget.
        river_obj (RiverTransect): RiverTransect object.
        mapbox_token (Union[bool, None], optional): mapbox token. This is needed
            to have the satellite image in the background. If None the plot will
            have a white background. Defaults to None.
        current_crs (str, optional): Current projection of the data. This is
            needed to reproject the data a WGS84 for mapbox plotting.
            Defaults to "epsg:4326".

    Returns:
        go.FigureWidget: Figure widget of the interactive plot.
    """
    # Check if River has meanders
    if len(river_obj.meanders) > 0:
        for meander_id in list(river_obj.meanders):
            hover_template = create_hover_template_meander(
                meander_id, river_obj
            )
            meander = river_obj.meanders[meander_id]
            x = meander.x
            y = meander.y
            if mapbox_token is not None:
                satellite = True
                if current_crs != "epsg:4326":
                    wgs84 = pyproj.CRS("EPSG:4326")
                    projected = pyproj.CRS(current_crs)
                    transformer = pyproj.Transformer.from_crs(projected, wgs84)
                    y_lat, x_lon = transformer.transform(x, y)
                else:
                    x_lon = copy.deepcopy(x)
                    y_lat = copy.deepcopy(y)
            else:
                satellite = False

            if satellite:
                f.add_trace(
                    go.Scattermapbox(
                        lon=[x_lon[0]],
                        lat=[y_lat[0]],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name=f"Meander {meander_id}",
                        showlegend=False,
                        hovertemplate=hover_template,
                    )
                )
                f.add_trace(
                    go.Scattermapbox(
                        lon=[x_lon[-1]],
                        lat=[y_lat[-1]],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name=f"Meander {meander_id}",
                        showlegend=False,
                        hovertemplate=hover_template,
                    )
                )

                f.add_trace(
                    go.Scattermapbox(
                        lon=x_lon,
                        lat=y_lat,
                        mode="lines",
                        line=dict(color="red", width=2),
                        name=f"Meander {meander_id}",
                        hovertemplate=hover_template,
                    )
                )
            else:
                f.add_trace(
                    go.Scatter(
                        x=[x[0]],
                        y=[y[0]],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name=f"Meander {meander_id}",
                        showlegend=False,
                        hovertemplate=hover_template,
                    )
                )
                f.add_trace(
                    go.Scatter(
                        x=[x[-1]],
                        y=[y[-1]],
                        mode="markers",
                        marker=dict(color="red", size=10),
                        name=f"Meander {meander_id}",
                        showlegend=False,
                        hovertemplate=hover_template,
                    )
                )

                f.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(color="red", width=2),
                        name=f"Meander {meander_id}",
                        hovertemplate=hover_template,
                    )
                )
    return f


def create_hover_template_meander(meander_id: int, river_obj=None) -> str:
    """create hover template for meanders.

    Args:
        meander_id (int): Meander ID
        river_obj (Union[None, RiverTransect], optional): RiverTransect object.
            Defaults to Union[None, RiverTransect].

    Returns:
        str: hover template.
    """
    hover_template = f"<b>Meander ID: {meander_id}</b>"
    if river_obj is not None:
        lambda_value = river_obj.meanders[meander_id].data["lambda_fm"]
        l_value = river_obj.meanders[meander_id].data["L_fm"]
        sinuosity_value = river_obj.meanders[meander_id].data["sigma_fm"]
        radius = river_obj.meanders[meander_id].data["R_hm"]
        hover_template += f"<br>Lambda_fm: {lambda_value:.2f} m"
        hover_template += f"<br>L_fm: {l_value:.2f} m"
        hover_template += f"<br>Sinuosity_fm: {sinuosity_value:.2f}"
        hover_template += f"<br>Radius_hm: {radius:.2f}"

    return hover_template
