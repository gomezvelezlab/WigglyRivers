# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#
#                               Last revised 2021-01-25
# _____________________________________________________________________________
# _____________________________________________________________________________
'''
The functions given on this package allow the user to save data in different
formats

'''
# ------------------------
# Importing Modules
# ------------------------ 
import os
import copy
# Data Managment
import geopandas as gpd
from shapely import LineString
import fiona
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
    return ['p', 'pickle', 'mat', 'json', 'txt', 'csv', 'shp', 'hdf5',
            'feather']


def get_load_formats():
    return ['p', 'pickle', 'mat', 'json', 'txt', 'csv', 'shp', 'dbf', 'hdf5',
            'feather']


def save_data(data, path_output, file_name, *args, **kwargs):
    """
    DESCRIPTION:
        Saves data depending on the format. It can save files in pickle,
        matlab, cvs, and txt.
    _______________________________________________________________________
    INPUT:
        :param data: dict,
            Dictionary with the data to be saved.
        :type data: dict or gpd.read_file() or pd.DataFrame
        :param path_output: str,
            Directory to be saved, the directory will be created.
        :type path_output: str
        :param file_name: str,
            Name of the file, it must include the extension.
        :type file_name: str
    _______________________________________________________________________
    OUTPUT:
        Saves the data.
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
    name_out = os.path.join(path_output,file_name) #f'{path_output}{file_name}'
    extension = file_name.split('.')[-1]

    dataframe = copy.deepcopy(data)
    if (isinstance(data, pd.DataFrame) or isinstance(
        data, gpd.GeoDataFrame)) and extension not in ('shp', 'txt',
                                                       'csv', 'feather'):
        data = {}
        for i in dataframe.columns:
            data[i] = dataframe[i].values
    
    if isinstance(data, dict) and extension in ('shp', 'txt', 'csv', 'feather'):
        dataframe = pd.DataFrame.from_dict(data)

    if extension == 'mat':
        sio.savemat(name_out, data, *args, **kwargs)
    elif extension in ('txt', 'csv'):
        # dataframe = pd.DataFrame.from_dict(data)
        dataframe.to_csv(name_out, *args, **kwargs)
    elif extension == 'feather':
        # dataframe = pd.DataFrame.from_dict(data)
        dataframe.to_feather(name_out, *args, **kwargs)
    elif extension in ('p', 'pickle'):
        file_open = open(name_out, "wb")
        pickle.dump(data, file_open)
        file_open.close()
    elif extension == 'json':
        with open(name_out, 'w') as json_file:
            json.dump(data, json_file, cls=NpEncoder)
    elif extension == 'shp':
        if isinstance(data, pd.DataFrame):
            data = gpd.GeoDataFrame(data, geometry=data.geometry)
        data.to_file(name_out)
    elif extension == 'hdf5':
        save_dict_to_hdf5(data, name_out)
    else:
        raise CE.FormatError(
            f'format .{extension} not implemented. '
            f'Use extensions {get_save_formats()}')


def load_data(file_data, pandas_dataframe=False, *args, **kwargs):
    """
    DESCRIPTION:
        Loads data depending on the format and returns a dictionary.

        The data can be loaded from pickle, matlab, csv, or txt.
    _______________________________________________________________________
    INPUT:
        :param file_data: str,
            Data file
        :param pandas_dataframe: boolean,
            If true returns a pandas dataframe instead of a dictionary.
                                
    _______________________________________________________________________
    OUTPUT:
        :return data: dict,
            Dictionary or pandas dataframe with the data in the file.
    """
    # ---------------------
    # Error Management
    # ---------------------
    if not isinstance(file_data, str):
        raise TypeError('data must be a string.')

    try:
        keys = kwargs['keys']
    except:
        keys = None

    # ---------------------
    # load data
    # ---------------------
    extension = file_data.split('.')[-1].lower()
    if extension == 'mat':
        data = sio.loadmat(file_data, *args, **kwargs)
    elif extension in ('txt', 'csv'):
        dataframe = pd.read_csv(file_data, *args, **kwargs)
        data = {}
        for i in dataframe.columns:
            data[i] = dataframe[i].values
    elif extension == 'feather':
        dataframe = pd.read_feather(file_data, *args, **kwargs)
        data = {}
        for i in dataframe.columns:
            data[i] = dataframe[i].values
    elif extension in ('p', 'pickle'):
        file_open = open(file_data, "rb")
        data = pickle.load(file_open)
        file_open.close()
    elif extension == 'json':
        with open(file_data) as f:
            data = json.load(f)
    elif extension == 'shp':
        data = gpd.read_file(file_data)
    elif extension == 'dbf':
        from simpledbf import Dbf5
        dbf = Dbf5(file_data)
        df = dbf.to_dataframe()
        data = {}
        for i in df.columns:
            data[i] = df[i].values
    elif extension == 'hdf5':
        data = load_dict_from_hdf5(file_data, key_c=keys)
        # with h5py.File(file_data, 'r') as f:
        #     if keys is None:
        #         keys = list(f.keys())
        #     data = {key: np.array(f[key]) for key in keys}
    else:
        raise CE.FormatError(
            f'format .{extension} not implemented. '
            f'Use files with extensions {get_load_formats()}')
    if pandas_dataframe:
        data = pd.DataFrame.from_dict(data)
    return data


def readGDB(file_data, layer):
    """
    DESCRIPTION:
        Loads data from a geodatabase (GDB).
    _______________________________________________________________________
    INPUT:
        :param file_data: str,
            Data file
        :param layer: str,
            Layer that will be loaded from the GDB.
    """
    layers = fiona.listlayers(file_data)
    try:
        layers.index(layer)
    except ValueError:
        raise KeyError(f'{layer} is not present in the GDB')
    shapefile = gpd.read_file(file_data, driver='FileGDB', layer=layer)
    return shapefile


def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    types = (np.ndarray, np.int64, np.float64, str, bytes, float, int, bool,
             list, np.bool_, np.int_, np.float_, np.str_, np.bytes_, np.int32,
             np.float32)
    for key, item in dic.items():
        if isinstance(item, types):
            try:
                h5file[path + str(key)] = item
            except ValueError:
                lengths = [len(i) for i in item]
                max_length = max(lengths)
                # create array with max length
                array = np.full((len(item), max_length), np.nan)
                # fill array with item
                for i in range(len(item)):
                    array[i, :lengths[i]] = item[i]
                h5file[path + str(key)] = array 

        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(
                h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename, key_c=None):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        dataset = recursively_load_dict_contents_from_group(
            h5file, '/', key_c=key_c)
        h5file.close()
        return dataset

def recursively_load_dict_contents_from_group(h5file, path, key_c=None):
    """
    ....
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
                ans[key] = ans[key].decode('utf-8')
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(
                h5file, path + key + '/')
    return ans


def create_geopandas_dataframe(pandas_df, geometry_columns, shape_type='line',
                               crs='EPSG:4326'):
    """
    Description:
        Creates a geopandas dataframe from a pandas dataframe.
    ____________________________________________________________________________

    Args:
    -----
    :param pandas_df: pandas dataframe,
        pandas dataframe with the data.
    :param geometry_columns: list,
        List of the columns that will be used to create the geometry.
    :param type_shape: str,
        Type of geometry that will be created. Options: 'line', 'point'.
    """
    # Create geometry
    if shape_type.lower() == 'point':
        geometry = gpd.points_from_xy(pandas_df[geometry_columns[0]],
                                      pandas_df[geometry_columns[1]])
    elif shape_type.lower() == 'line':
        geometry = [LineString(np.array(xy).T)
                    for xy in zip(pandas_df[geometry_columns[0]].values,
                                  pandas_df[geometry_columns[1]].values)]
    else:
        raise ValueError(f'{shape_type} is not a valid type of geometry.')
    
    # Create geopandas dataframe
    gdf = gpd.GeoDataFrame(pandas_df, crs=crs, geometry=geometry)

    # Remove object columns
    for i in gdf.columns:
        if gdf[i].dtype == 'object':
            gdf = gdf.drop(i, axis=1)

    return gdf

def read_gbd(file_data, layer):
    """
    DESCRIPTION:
        Loads data from a geodatabase (GDB).
    _______________________________________________________________________
    INPUT:
        :param file_data: str,
            Data file
        :param layer: str,
            Layer that will be loaded from the GDB.
    """
    shapefile = gpd.read_file(file_data, driver='FileGDB', layer=layer)
    return shapefile
