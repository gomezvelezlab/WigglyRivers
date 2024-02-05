
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
overwrite = False
# Parameters
huc_id = '0602'
# huc_id = '0104'
# huc_id = '0601'
# huc_id = '0513'
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
range_values = np.append(initial_range, range_values[1:])
if range_values[-1] < full_dataset:
    range_values = np.append(range_values, full_dataset)
elif range_values[-1] > full_dataset:
    range_values[-1] = full_dataset

print(f'Number of folders to save {len(range_values) - 1}') 
# TODO: This is a good example of within waterbody in 0513!
# range = [1007, 1008]
time_all = time.time()
for r_val, i in enumerate(range_values[:-1]):
    # Start saving data variables
    cwt_data = {}
    reach_metrics_downstream = {}
    reach_metrics_no_clip = {}
    meander_database_downstream = {}
    # Select range
    range = [range_values[r_val], range_values[r_val + 1]]
    # range = [1007, 1008]

    # Check if folder is already extracted
    if range[1] - range[0] > 1:
        path_output_meanders = f'{path_meander_info}/{range[0]}_{range[1]-1}/'
    else:
        path_output_meanders = f'{path_meander_info}/{range[0]}/'
    if os.path.exists(f'{path_output_meanders}reach_metrics_no_clip.hdf5') and not overwrite:
        print(f'Folder {path_output_meanders} already exists')
        continue
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
    flag_reach = True
    for i_val, id_river in enumerate(id_rivers):
        # Test River
        # if i_val != 28:
        #     continue
        if i_val % 500 == 0:
            print(f'Processing River {i_val} of {len(id_rivers)}')
            utl.toc(time1)
        river = rivers[id_river]
        # -----------------------------
        # Calculate Curvature
        # -----------------------------
        rivers[id_river].calculate_curvature(data_source='resample') 
        # ---------------------------
        # Calculate CWT with Morlet
        # ---------------------------
        rivers[id_river].get_cwt_curvature(mother='MORLET')
        rivers[id_river].get_cwt_angle(mother='MORLET')
        cwt_morlet = rivers.extract_cwt_data(rivers_ids=[id_river])
        rivers[id_river].get_cwt_curvature(mother='DOG')
        rivers[id_river].get_cwt_angle(mother='DOG')
        cwt_dog = rivers.extract_cwt_data(rivers_ids=[id_river])
        cwt_data[id_river] = {'morlet': cwt_morlet, 'dog': cwt_dog}
        # -----------------------------
        # Extract CWT tree
        # -----------------------------
        try:
            rivers[id_river].extract_cwt_tree()
        except ValueError:
            # Some areas are so small that the cwt tree cannot be extracted
            continue

        # Test if tree scales is not None
        if rivers[id_river].tree_scales is None:
            continue
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
        # With clip downstream
        rivers[id_river].add_meanders_from_tree_scales(
            bounds_array_str='inflection', overwrite=True, clip='downstream')
        # ---------------------------
        # Calculate reach sinuosity
        # ---------------------------
        rivers[id_river].calculate_reach_metrics()
        # Extract data
        metrics_reach = rivers[id_river].metrics_reach
        reach_metrics_downstream[id_river] = metrics_reach
        meander_database = rivers[id_river].database
        if flag_reach:
            meander_database_downstream = copy.deepcopy(meander_database)
            flag_reach = False
        else:
            meander_database_downstream = pd.concat([
                meander_database_downstream, meander_database])
        # -----------------------------
        # Add meander to database
        # -----------------------------
        # TODO: Claculate reach metrics with both, the clip downstream and no
        #  clip and save them separately
        # TODO: Found a bug while extracting meanders with extended bounds.
        #  See 0602 - i_val = 28, tree_id = 3, node.node_id = 1
        rivers[id_river].add_meanders_from_tree_scales(
            bounds_array_str='extended', overwrite=True, clip='no')
        # ---------------------------
        # Calculate reach sinuosity
        # ---------------------------
        rivers[id_river].calculate_reach_metrics()
        metrics_reach = rivers[id_river].metrics_reach
        reach_metrics_no_clip[id_river] = metrics_reach

    utl.toc(time1)
    # Save rivers
    print('Saving Information')
    FM.save_data(cwt_data, path_output_meanders, 'cwt_data.hdf5')
    rivers.save_databases_meanders(
        path_output_meanders, f'meander_database.feather')
    rivers.save_databases_meanders(
        path_output_meanders, f'meander_database.csv')
    FM.save_data(meander_database_downstream, path_output_meanders,
                 'meander_database_downstream.feather')
    FM.save_data(reach_metrics_downstream, path_output_meanders,
                 'reach_metrics_downstream.hdf5')
    FM.save_data(reach_metrics_no_clip, path_output_meanders,
                 'reach_metrics_no_clip.hdf5')
    rivers.save_tree_scales(path_output_meanders)
    # rivers.save_rivers(
    #     path_output_meanders, file_name=f'rivers.hdf5',
    #     fn_meander_database='meander_database.feather',
    #     save_cwt_info=False, rivers_ids=id_rivers_extracted)
    print('Information Saved')
utl.toc(time_all)