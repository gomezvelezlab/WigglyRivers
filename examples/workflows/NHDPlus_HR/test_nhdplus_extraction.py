
# Importing Packages
import os
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

# Importing WigglyRivers Packages
from WigglyRivers import RiverDatasets, RiverTransect
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
huc_id = '0602'
# huc_id = '0104'
# huc_id = '0601'
path_nhd = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/raw_data/NHDPlus_H_{huc_id}_HU4_GDB/'
path_output = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/processed_data/NHDPlus_H_{huc_id}_HU4_GDB/'
path_river_routing = f'{path_output}/river_routing/'
path_meander_info = f'{path_output}/meanders/'
file_name = f'NHDPlus_H_{huc_id}_HU4_GDB.gdb'
projection = 'esri:102003'

# ------------------------------
# Start Logger
# ------------------------------
logger = Logger(console=True)
# ------------------------------
# Extract NHD Information
# ------------------------------
# nhd = NHD(path_output=path_output, logger=logger, save_format='feather')
# nhd.get_data_from_nhd_gbd(f'{path_nhd}{file_name}', projection=projection)
# ------------------------------
# Arguments for the noise data
# ------------------------------
kwargs_resample = {
    '0':{'smooth': 1e2},
    }

# ------------------------------
# Create Rivers and Add Files
# ------------------------------
# Directories of the information
path_tables = f'{path_output}/tables/'
path_coords = f'{path_output}/coordinates/'

# Create Rivers object
# rivers = RiverDatasets(logger=logger)
rivers = RiverDatasets()
# Add  files
rivers.add_files(path_data=path_tables, huc04=huc_id, path_coords=path_coords,
                comid_id='nhdplusid', load_coords=True)
    
# Map the reach network
# logger.info('Mapping Network')
rivers.map_network(method='upstream', huc=4, path_out=path_river_routing)
    
# Load coordinates
linking_network_file = f'{path_river_routing}/linking_network.feather'
comid_network_file = f'{path_river_routing}/comid_network.hdf5'
rivers.load_linking_network(linking_network_file)
huc_list = rivers.load_huc_list_in_comid_network(comid_network_file)
huc = huc_list[0]
headwaters = rivers.load_extracted_in_comid_network(
    comid_network_file, huc=huc)
headwaters_to_extract = headwaters[huc][:300]

# ---------------------------
# Get Reaches from Network
# ---------------------------
rivers.get_reaches_from_network(
    huc=huc,
    headwaters_comid=headwaters_to_extract,
    linking_network_file=linking_network_file,
    min_distance=100.0, path_out=path_river_routing)



# ---------------------------
# Load Network
# ---------------------------
river_network_file = f'{path_river_routing}/river_network_huc_{huc}.hdf5'
rivers.load_river_network(river_network_file)
starting_comids = rivers.id_values

fig = rivers.plot_rivers(engine='plotly')
fig.show()

print('Done!')