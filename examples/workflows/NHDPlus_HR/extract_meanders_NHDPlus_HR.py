
# Importing Packages
import os
import time
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display, HTML
from IPython.display import display
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from scipy import interpolate
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Importing WigglyRivers Packages
from WigglyRivers import RiverDatasets as Rivers
from WigglyRivers import RiverFunctions as RF
from WigglyRivers import ExtractNHDPlusHRData as NHD
from WigglyRivers.utilities import utilities as utl
from WigglyRivers.utilities import filesManagement as FM
from WigglyRivers.utilities import graphs
from WigglyRivers import Logger
from WigglyRivers import WaveletTreeFunctions as WTFunc
from WigglyRivers import iwavelet

# -------------------------
# Parameters
# -------------------------
# Parameters
# huc_id = '0602'
# huc_id = '0104'
# huc_id = '0601'
huc_id = '0513'
path_nhd = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/raw_data/NHDPlus_H_{huc_id}_HU4_GDB/'
path_output = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/processed_data/NHDPlus_H_{huc_id}_HU4_GDB/'
path_river_routing = f'{path_output}/river_routing/'
path_meander_info = f'{path_output}/meanders/'
file_name = f'NHDPlus_H_{huc_id}_HU4_GDB.gdb'
projection = 'esri:102003'

# ------------------------------
# Start Logger
# ------------------------------
logger = Logger(console=False)


# -----------------------------
# Information
# -----------------------------
path_tables = f'{path_output}/tables/'
path_coords = f'{path_output}/coordinates/'
# -----------------------------
# Create Rivers object
# -----------------------------
rivers = Rivers(logger=logger)
# -----------------------------
# Add files
# -----------------------------
rivers.add_files(path_data=path_tables, huc04=huc_id, path_coords=path_coords,
                comid_id='nhdplusid', load_coords=False)
# -----------------------------
# load Linking Files
# -----------------------------
linking_network_file = f'{path_river_routing}/linking_network.feather'
comid_network_file = f'{path_river_routing}/comid_network.hdf5'
rivers.load_linking_network(linking_network_file)
# -----------------------------
# load HUC files extracted
# -----------------------------
huc_list = rivers.load_huc_list_in_comid_network(comid_network_file)
huc = huc_list[0]
headwaters = rivers.load_extracted_in_comid_network(
    comid_network_file, huc=huc)
headwaters = headwaters[huc]

# ====================================
# Iterate over series river transects
# ====================================
initial_step = 100
step = 1000
full_dataset = len(headwaters)
initial_range = np.array([0, initial_step])
range_values = np.arange(initial_range[-1], full_dataset, step)
# range_values = np.append(initial_range, range_values[1:])
if range_values[-1] < full_dataset:
    range_values = np.append(range_values, full_dataset)
elif range_values[-1] > full_dataset:
    range_values[-1] = full_dataset

print(len(range_values))
# TODO: This is a good example of within waterbody!
# range = [1007, 1008]
for r_val, i in enumerate(range_values[:-1]):
    # Select range
    range = [range_values[r_val], range_values[r_val + 1]]
    # -----------------------------
    # Create Rivers object
    # -----------------------------
    rivers = Rivers(logger=logger)
    # -----------------------------
    # Add files
    # -----------------------------
    rivers.add_files(path_data=path_tables, huc04=huc_id, path_coords=path_coords,
                    comid_id='nhdplusid', load_coords=False)
    # -----------------------------
    # load Linking Files
    # -----------------------------
    linking_network_file = f'{path_river_routing}/linking_network.feather'
    comid_network_file = f'{path_river_routing}/comid_network.hdf5'
    rivers.load_linking_network(linking_network_file)
    # -----------------------------
    # load HUC files extracted
    # -----------------------------
    huc_list = rivers.load_huc_list_in_comid_network(comid_network_file)
    huc = huc_list[0]
    headwaters = rivers.load_extracted_in_comid_network(
        comid_network_file, huc=huc)
    headwaters = headwaters[huc]
    # -----------------------------
    # Select River
    # -----------------------------
    headwaters = headwaters[range[0]: range[1]]
    # -----------------------------
    # Load headwaters
    # -----------------------------
    river_network_file = f'{path_river_routing}/river_network_huc_{huc}.hdf5'
    kwargs_resample = {hw: {'smooth': 1e1, 'method': 'geometric_mean_width'} for hw in headwaters}
    rivers.load_river_network(river_network_file, headwaters_comid=headwaters,
                            kwargs_resample=kwargs_resample)
    id_rivers = rivers.id_values
    print(f'Rivers to process: {len(id_rivers)}')
    if len(id_rivers) == 0:
        continue

    id_rivers_extracted = id_rivers
    time1 = time.time()
    for i_val, id_river in enumerate(id_rivers):
        if i_val % 500 == 0:
            print(f'Processing River {i_val} of {len(id_rivers)}')
        river = rivers[id_river]
        # -----------------------------
        # Calculate Curvature
        # -----------------------------
        rivers[id_river].calculate_curvature(data_source='resample') 
        # -----------------------------
        # Extract CWT tree
        # -----------------------------
        rivers[id_river].extract_cwt_tree()
        # -----------------------------
        # Prune by peak power
        # -----------------------------
        rivers[id_river].prune_tree_by_peak_power()
        # -----------------------------
        # Prune by sinuosit
        # -----------------------------
        rivers[id_river].prune_tree_by_sinuosity(1.01)
        # -----------------------------
        # Add meander to database
        # -----------------------------
        rivers[id_river].add_meanders_from_tree_scales(
            bounds_array_str='extended')
        # ---------------------------
        # Calculate reach sinuosity
        # ---------------------------
        rivers[id_river].calculate_reach_metrics()
    utl.toc(time1)

    # Save rivers
    print('Saving Information')
    if range[1] - range[0] > 1:
        path_output_meanders = f'{path_meander_info}/{range[0]}_{range[1]-1}/'
    else:
        path_output_meanders = f'{path_meander_info}/{range[0]}/'
    rivers.save_databases_meanders(
        path_output_meanders, f'meander_database.csv')
    rivers.save_rivers(
        path_output_meanders, file_name=f'rivers.hdf5',
        fn_meander_database='meander_database.feather',
        save_cwt_info=False, rivers_ids=id_rivers_extracted)
