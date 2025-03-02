# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#
#                               Last revised 2025-02-13
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
The functions given on this package allow the user to save data in different
formats

"""
# ------------------------
# Importing Modules
# ------------------------
import os
import copy
from pathlib import Path
from typing import Union

# Data Managment
import geopandas as gpd
from shapely import LineString
import pickle
import scipy.io as sio
import pandas as pd

# import netCDF4 as nc
import json
import numpy as np
import h5py

# Personal libaries
from . import utilities as utl
from . import classExceptions as CE


# ------------------------
# Functions
# ------------------------
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_save_formats():
    return [
        "p",
        "pickle",
        "mat",
        "json",
        "txt",
        "csv",
        "shp",
        "hdf5",
        "feather",
    ]


def get_load_formats():
    return [
        "p",
        "pickle",
        "mat",
        "json",
        "txt",
        "csv",
        "shp",
        "dbf",
        "hdf5",
        "feather",
    ]


def save_data(
    data: dict, path_output: Union[Path, str], file_name: str, *args, **kwargs
):
    """Save data in any of the following formats:
        .p, .pickle, .mat, .json, .txt, .csv, .shp, .hdf5, .feather

    Args:
        data (dict): Data to be saved in dictionary format.
        path_output (Union[Path, str]): Path where the data will be saved.
        file_name (str): File name.
        *args: Additional arguments for each saving function.
        **kwargs: Additional keyword arguments for each saving function.

    Raises:
        CE.FormatError: Format not implemented.
    """
    # ---------------------
    # Error Management
    # ---------------------
    # if not isinstance(data, dict) and not isinstance(
    #         data, gpd.geodataframe.GeoDataFrame):
    #     raise TypeError('data must be a dictionary or a geopandas dataframe')
    # ---------------------
    # Create Folder
    # ---------------------
    utl.cr_folder(path_output)

    # ---------------------
    # Save data
    # ---------------------
    name_out = os.path.join(
        path_output, file_name
    )  # f'{path_output}{file_name}'
    extension = file_name.split(".")[-1]

    dataframe = copy.deepcopy(data)
    if (
        isinstance(data, pd.DataFrame) or isinstance(data, gpd.GeoDataFrame)
    ) and extension not in ("shp", "txt", "csv", "feather"):
        data = {}
        for i in dataframe.columns:
            data[i] = dataframe[i].values

    if isinstance(data, dict) and extension in ("shp", "txt", "csv", "feather"):
        dataframe = pd.DataFrame.from_dict(data)

    if extension == "mat":
        sio.savemat(name_out, data, *args, **kwargs)
    elif extension in ("txt", "csv"):
        # dataframe = pd.DataFrame.from_dict(data)
        dataframe.to_csv(name_out, *args, **kwargs)
    elif extension == "feather":
        # dataframe = pd.DataFrame.from_dict(data)
        dataframe.to_feather(name_out, *args, **kwargs)
    elif extension in ("p", "pickle"):
        file_open = open(name_out, "wb")
        pickle.dump(data, file_open)
        file_open.close()
    elif extension == "json":
        with open(name_out, "w") as json_file:
            json.dump(data, json_file, cls=NpEncoder)
    elif extension == "shp":
        if isinstance(data, pd.DataFrame):
            data = gpd.GeoDataFrame(data, geometry=data.geometry)
        data.to_file(name_out)
    elif extension == "hdf5":
        save_dict_to_hdf5(data, name_out)
    else:
        raise CE.FormatError(
            f"format .{extension} not implemented. "
            f"Use extensions {get_save_formats()}"
        )


def load_data(
    file_path: Union[Path, str], pandas_dataframe: bool = False, *args, **kwargs
) -> Union[dict, pd.DataFrame]:
    """Load data in any of the following formats:
        .p, .pickle, .mat, .json, .txt, .csv, .shp, .dbf, .hdf5, .feather


    Args:
        file_path (Union[Path, str]): file to be loaded.
        pandas_dataframe (bool, optional): Load the data as PandasDataframe.
            If False, the data will be loaded as a dictionary. Defaults to False.
        *args: Additional arguments for each loading function.
        **kwargs: Additional keyword arguments for each loading function.

    Raises:
        TypeError: file_path must be a string or Path.
        CE.FormatError: format not implemented.

    Returns:
        Union[dict, pd.DataFrame]: Loaded data.
    """

    # ---------------------
    # Error Management
    # ---------------------
    if not isinstance(file_path, (str, Path)):
        raise TypeError("data must be a string.")

    try:
        keys = kwargs["keys"]
    except KeyError:
        keys = None

    # ---------------------
    # load data
    # ---------------------
    file_path = str(file_path)
    extension = file_path.split(".")[-1].lower()
    if extension == "mat":
        data = sio.loadmat(file_path, *args, **kwargs)
    elif extension in ("txt", "csv"):
        dataframe = pd.read_csv(file_path, *args, **kwargs)
        data = {}
        for i in dataframe.columns:
            data[i] = dataframe[i].values
    elif extension == "feather":
        dataframe = pd.read_feather(file_path, *args, **kwargs)
        data = {}
        for i in dataframe.columns:
            data[i] = dataframe[i].values
    elif extension in ("p", "pickle"):
        file_open = open(file_path, "rb")
        data = pickle.load(file_open)
        file_open.close()
    elif extension == "json":
        with open(file_path) as f:
            data = json.load(f)
    elif extension == "shp":
        data = gpd.read_file(file_path)
    elif extension == "dbf":
        from simpledbf import Dbf5

        dbf = Dbf5(file_path)
        df = dbf.to_dataframe()
        data = {}
        for i in df.columns:
            data[i] = df[i].values
    elif extension == "hdf5":
        data = load_dict_from_hdf5(file_path, key_c=keys)
    else:
        raise CE.FormatError(
            f"format .{extension} not implemented. "
            f"Use files with extensions {get_load_formats()}"
        )
    if pandas_dataframe:
        data = pd.DataFrame.from_dict(data)
    return data


def save_dict_to_hdf5(dic: dict, file_name: str):
    """save dictionary to hdf5 file

    Args:
        dic (dict): Dictionary to be saved.
        file_name (str): Complete path to the file.
    """
    with h5py.File(file_name, "w") as h5file:
        recursively_save_dict_contents_to_group(h5file, "/", dic)


def recursively_save_dict_contents_to_group(
    h5file: h5py.File, path: str, dic: dict
):
    """recursively save dictionary to hdf5 file

    Args:
        h5file (h5py.File): h5py file object.
        path (str): path to variable.
        dic (dict): dictionary to be saved.

    Raises:
        ValueError: save type not implemented.
    """
    types = (
        np.ndarray,
        np.int64,
        np.float64,
        str,
        bytes,
        float,
        int,
        bool,
        list,
        np.bool_,
        np.int_,
        np.float_,
        np.str_,
        np.bytes_,
        np.int32,
        np.float32,
    )
    for key, item in dic.items():
        if isinstance(item, types):
            try:
                # Handle string data types
                if isinstance(item, str):
                    h5file[path + str(key)] = np.string_(item)
                elif (
                    isinstance(item, (list, np.ndarray))
                    and len(item) > 0
                    and isinstance(item[0], str)
                ):
                    string_array = np.array([np.string_(s) for s in item])
                    h5file[path + str(key)] = string_array
                else:
                    h5file[path + str(key)] = item
            except ValueError:
                lengths = [len(i) for i in item]
                max_length = max(lengths)
                # create array with max length
                array = np.full((len(item), max_length), np.nan)
                # fill array with item
                for i in range(len(item)):
                    array[i, : lengths[i]] = item[i]
                h5file[path + str(key)] = array
            except TypeError:
                if isinstance(item, (list, np.ndarray)):
                    # Convert all items to strings and then to bytes
                    item2 = [np.string_(str(i)) for i in item]
                    h5file[path + str(key)] = item2
                else:
                    # Single item case
                    h5file[path + str(key)] = np.string_(str(item))
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(
                h5file, path + key + "/", item
            )
        else:
            raise ValueError("Cannot save %s type" % type(item))


def load_dict_from_hdf5(
    file_name: Union[Path, str], key_c: Union[str, None] = None
) -> dict:
    """load dictionary from hdf5 file.

    Args:
        file_name (Union[Path, str]): complete path to the file.
        key_c (Union[str, None], optional): specific key to be loaded from the
            hdf5 file. If key_c is None, all the keys will be loaded.
            Defaults to None.

    Returns:
        dict: loaded dictionary.
    """
    with h5py.File(file_name, "r") as h5file:
        dataset = recursively_load_dict_contents_from_group(
            h5file, "/", key_c=key_c
        )
        h5file.close()
        return dataset


def recursively_load_dict_contents_from_group(
    h5file: h5py.File, path: str, key_c: Union[str, None] = None
) -> dict:
    """recursively load dictionary from hdf5 file.

    Args:
        h5file (h5py.File): h5py file object.
        path (str): path to variable.
        key_c (Union[str, None], optional): specific key to be loaded from the
            hdf5 file. If key_c is None, all the keys will be loaded.
            Defaults to None.

    Returns:
        dict: Loaded dictionary
    """
    ans = {}
    for key, item in h5file[path].items():
        if key_c is not None:
            if isinstance(key_c, str):
                key_c = [key_c]
            if key not in key_c:
                continue
        if isinstance(item, h5py._hl.dataset.Dataset):
            # ans[key] = item.value
            try:
                ans[key] = item[:]
            except ValueError:
                try:
                    ans[key] = item.value
                except AttributeError:
                    ans[key] = item[()]
            if isinstance(ans[key], bytes):
                ans[key] = ans[key].decode("utf-8")
            if isinstance(ans[key], np.ndarray):
                if isinstance(ans[key][0], bytes):
                    ans[key] = np.array([i.decode("utf-8") for i in ans[key]])
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + "/"
            )
    return ans


def create_geopandas_dataframe(
    pandas_df: pd.DataFrame,
    geometry_columns: list,
    shape_type: str = "line",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Create geopandas dataframe from a pandas dataframe.

    Args:
        pandas_df (pd.DataFrame): Pandas dataframe.
        geometry_columns (list): list of columns that hold the geometry.
        shape_type (str, optional): geometry type. The options are "point" or
            "line". Defaults to "line".
        crs (str, optional): projection. Defaults to "EPSG:4326".

    Raises:
        ValueError: geometry type not implemented.

    Returns:
        gpd.GeoDataFrame: geopandas dataframe.
    """

    # Create geometry
    if shape_type.lower() == "point":
        geometry = gpd.points_from_xy(
            pandas_df[geometry_columns[0]], pandas_df[geometry_columns[1]]
        )
    elif shape_type.lower() == "line":
        geometry = [
            LineString(np.array(xy).T)
            for xy in zip(
                pandas_df[geometry_columns[0]].values,
                pandas_df[geometry_columns[1]].values,
            )
        ]
    else:
        raise ValueError(f"{shape_type} is not a valid type of geometry.")

    # Create geopandas dataframe
    gdf = gpd.GeoDataFrame(pandas_df, crs=crs, geometry=geometry)

    # Remove object columns
    for i in gdf.columns:
        if gdf[i].dtype == "object":
            gdf = gdf.drop(i, axis=1)

    return gdf


def read_gdb(
    file_path: Union[Path, str], layer: str, **kwargs
) -> gpd.GeoDataFrame:
    """Read a layer from a geodatabase.

    Args:
        file_path (Union[Path, str]): Path to the geodatabase.
        layer (str): layer to be read.

    Returns:
        gpd.GeoDataFrame: geopandas dataframe.
    """

    shapefile = gpd.read_file(
        file_path, driver="FileGDB", layer=layer, **kwargs
    )
    return shapefile
