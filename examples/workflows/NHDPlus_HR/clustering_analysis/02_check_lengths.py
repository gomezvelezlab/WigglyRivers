
# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#
#                               Last revised 2024-02-12
# _____________________________________________________________________________
# _____________________________________________________________________________
# Importing Packages
import os
import time
import copy
import numpy as np
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
import statsmodels.api as sm
import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

# Importing WigglyRivers Packages
from WigglyRivers import RiverDatasets as Rivers
from WigglyRivers import RiverFunctions as RF
from WigglyRivers import ReachExtraction as RE
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
path_data = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/WBD/compiled_data/'
# path_river_routing = f'{path_output}/river_routing/'
# path_meander_info = f'{path_output}/meanders/'
# file_name = f'NHDPlus_H_{huc_id}_HU4_GDB.gdb'
# projection = 'esri:102003'
# =========================
# Load lengths
# =========================
lengths_huc4_df = FM.load_data(f'{path_data}lengths_hw_huc4.feather',
                               pandas_dataframe=True)

lengths_huc6_df = FM.load_data(f'{path_data}lengths_hw_huc6.feather',
                               pandas_dataframe=True)

lengths_huc8_df = FM.load_data(f'{path_data}lengths_hw_huc8.feather',
                               pandas_dataframe=True)

# ======================================================
# Plot results
# ======================================================
# plot the headwaters
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
ax = axs[0]
ax.set_title(f'HUC4 N={len(lengths_huc4_df)}')
sns.histplot(lengths_huc4_df['length']/1000, ax=ax, color='skyblue')
ax.set_xlabel('Length (km)')
# ax.set_xscale('log')
# Plot histogram of lengths
ax = axs[1]
ax.set_title(f'HUC6 N={len(lengths_huc6_df)}')
sns.histplot(lengths_huc6_df['length']/1000, ax=ax, color='skyblue')
ax.set_xlabel('Length (km)')
# ax.set_xscale('log')
# Plot histogram of lengths
ax = axs[2]
ax.set_title(f'HUC8 N={len(lengths_huc8_df)}')
sns.histplot(lengths_huc8_df['length']/1000, ax=ax, color='skyblue')
ax.set_xlabel('Length (km)')
# ax.set_xscale('log')

plt.tight_layout()

# rivers_hw_all.loc[comid_network].plot(ax=ax, alpha=1, color='b', linewidth=1)
# rivers_hw_reachcode.plot(ax=ax, alpha=0.5, column='StreamLeve', cmap='YlOrRd') 
plt.savefig(f'test_huc_06.png')
plt.show()

print('done')
