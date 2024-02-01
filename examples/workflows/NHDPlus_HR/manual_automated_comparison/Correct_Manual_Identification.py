# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#                               Last revised 2024-01-24
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
______________________________________________________________________________

 DESCRIPTION:
   This script corrects the manual selection
______________________________________________________________________________
"""
# ----------------
# Import packages
# ----------------
# System management
# Importing Packages
import os
import time
import copy
import numpy as np
import pyproj
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from joblib import Parallel, delayed
import plotly.io as pio
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean

# Importing pyMeander Packages
from WigglyRivers import RiverDatasets
from WigglyRivers import RiverFunctions as RF
from WigglyRivers import ExtractNHDPlusHRData as NHD
from WigglyRivers.utilities import utilities as utl
from WigglyRivers.utilities import filesManagement as FM
from WigglyRivers import Logger
from WigglyRivers import WaveletTreeFunctions as WTFunc


# ---------------------
# Functions
# ---------------------
def convert_str_float_list_vector(x_val):
    x_val = x_val.replace('[', '').replace(']', '').replace('\n', ',').replace(
        ' ', ',').split(',')
    x_val = np.array([float(x) for x in x_val if x != ''])
    return  x_val

# ---------------------
# Start Logger
# ---------------------
logger = Logger(console=True)


# ------------------
# Parameters
# ------------------
# Parameters
path_projects = 'examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/manual/'
projects = utl.get_folders(path_projects)
path_projects_all = [f'{path_projects}{p}/' for p in projects]
path_projects_c = 'examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/manual_resampled/'
path_projects_out = [f'{path_projects_c}{p}/' for p in projects]

print(projects)

# Current projection of information
projection = 'esri:102003'

# ------------------------
# Calculate automatic
# ------------------------
# bounds_array_str = 'inflection'
# Select project to load
for project in projects:
    print(f'Processing project {project}')
    try:
        i_p = projects.index(project)
    except ValueError:
        raise ValueError(f'Project "{project}" not found')
    project_to_load = path_projects_all[i_p]
    print('Set project to load: ' + project_to_load)
    rivers = RiverDatasets(logger=logger)
    river_network_file = f'{project_to_load}/rivers.hdf5'

    # Create kwargs for resampling
    kwargs_resample = {f'{project}_0': {'smooth': 1e2}}

    # Load River Network with the smooth coordinates
    rivers.load_river_network(
        river_network_file,
        fn_meanders_database=f'{project_to_load}/meander_database.csv')
    id_rivers = rivers.id_values

    # Create kwargs for resampling
    kwargs_resample = {f'{id_rivers[0]}':
                       {'smooth': 1e0, 'method': 'geometric_mean_width'}}
    # Load River Network to use the resampled coordinates
    rivers_c = RiverDatasets(logger=logger)
    rivers_c.load_river_network(
        river_network_file, kwargs_resample=kwargs_resample)

    # Select River
    id_river = id_rivers[0]
    river = rivers[id_river]
    river_c = rivers_c[id_river]

    # Update data source
    river_c.data_source = 'resample'
    # Reclaclulate curvature
    river_c.calculate_curvature()
    # Plot rivers
    # plt.figure()
    # plt.plot(river.x_o, river.y_o, 'k', label='Original')
    # plt.plot(river_c.x, river_c.y, 'b', label='Resampled')
    # plt.plot(river.x_smooth, river.y_smooth, 'r', label='Smoothed')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.legend()
    # plt.show()
    # ------------------------
    # Translate meanders
    # ------------------------
    # Get meander ids
    meander_ids = river.id_meanders
    # Go through the meanders and link them to the resample river coordinates
    #  instead of the smooth ones.
    x = river_c.x
    y = river_c.y
    for id_m in meander_ids:
        meander = river.meanders[id_m]
        x_m = meander.x
        y_m = meander.y
        # Find closest point to the starting and ending point of the meander
        #  in the resampled river coordinates
        dist = np.sqrt((x_m[0] - x)**2 + (y_m[0] - y)**2)
        i_min = np.argmin(dist)
        dist = np.sqrt((x_m[-1] - x)**2 + (y_m[-1] - y)**2)
        i_max = np.argmin(dist)
        # Add meander into the river_c class
        river_c.add_meander(id_m, i_min, i_max)
        # river_c.meanders[id_m].plot_meander()
        # river.meanders[id_m].plot_meander()
    
    # ------------------------
    # Save river
    # ------------------------
    rivers_c.save_rivers(
        path_output=f'{path_projects_c}{project}/',
        file_name='rivers_manual.hdf5',
        fn_meander_database='meander_database_manual.csv')
    rivers_c.save_rivers(
        path_output=f'{path_projects_c}{project}/',
        file_name='rivers_manual.hdf5',
        fn_meander_database='meander_database_manual.feather')
    
print('Done')