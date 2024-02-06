# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#                               Last revised 2023-08-15
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
______________________________________________________________________________

 DESCRIPTION:
   This script compares the manual and the automatic extraction of the
   meanders
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
from IPython.display import Image, display, HTML
from joblib import Parallel, delayed
import plotly.io as pio
import ipywidgets as widgets
from IPython.display import display
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
path_projects = f'examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/manual_corrected/'
projects = utl.get_folders(path_projects)
projects = list(np.sort(projects))
path_projects_all = [f'{path_projects}{p}/' for p in projects]
path_projects_c = f'examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/manual_with_automatic/'
path_projects_out = [f'{path_projects_c}{p}/' for p in projects]
print(projects)
# Current projection of information
projection = 'esri:102003'


# ------------------------
# Calculate automatic
# ------------------------
bounds_array_str = 'extended'
clip = ''
# Select project to load
for project in projects:
    time_1 = time.time()
    try:
        i_p = projects.index(project)
    except ValueError:
        raise ValueError(f'Project "{project}" not found')

    project_to_load = path_projects_all[i_p]
    print('Set project to load: ' + project_to_load)
    rivers = RiverDatasets(logger=logger)
    river_network_file = f'{project_to_load}/rivers_manual_corrected.hdf5'
    # Load corrected files
    try:
        rivers.load_river_network(
            river_network_file,
            fn_meanders_database=f'{project_to_load}/meander_database_manual_corrected.feather')
    except FileNotFoundError:
        river_network_file = f'{project_to_load}/rivers_manual.hdf5'
        rivers.load_river_network(
            river_network_file,
            fn_meanders_database=f'{project_to_load}/meander_database_manual.feather')
    id_rivers = rivers.id_values

    # Select River
    id_river = id_rivers[0]
    river = rivers[id_river]
    # -----------------------------
    # Calculate Smooth Coordinates
    # -----------------------------
    # rivers[id_river].calculate_smooth(poly_order=2, gaussian_window=1,)
    # -----------------------------
    # Calculate Curvature
    # -----------------------------
    rivers[id_river].calculate_curvature() 
    # -----------------------------
    # Extract CWT tree
    # -----------------------------
    rivers[id_river].extract_cwt_tree()
    utl.toc(time_1)
    # -----------------------------
    # Prune by peak power
    # -----------------------------
    rivers[id_river].prune_tree_by_peak_power()
    # -----------------------------
    # Prune by sinuosity
    # -----------------------------
    # rivers[id_river].prune_tree_by_sinuosity(1.05)
    # -----------------------------
    # Add meander to database
    # -----------------------------
    rivers[id_river].add_meanders_from_tree_scales(
        bounds_array_str=bounds_array_str, clip='no')
    # -----------------------------
    # Save database
    # -----------------------------
    rivers.save_rivers(
        path_output=f'{path_projects_c}{project}/',
        file_name='rivers_automatic.hdf5',
        fn_meander_database=f'meander_database_{bounds_array_str}{clip}.csv')
    rivers.save_rivers(
        path_output=f'{path_projects_c}{project}/',
        file_name='rivers_automatic.hdf5',
        fn_meander_database=f'meander_database_{bounds_array_str}_clip_{clip}.feather')
    utl.toc(time_1)



