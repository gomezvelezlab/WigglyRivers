
# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#
#                               Last revised 2024-02-12
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
Description:
This script is intended to clip the river transects to the
HUC4, HUC6, HUC8, and HUC10. We decided to pick 100 km as our mark for the
clipping.
"""
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
path_out = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/WBD/cliped_data/'
# path_river_routing = f'{path_output}/river_routing/'
# path_meander_info = f'{path_output}/meanders/'
# file_name = f'NHDPlus_H_{huc_id}_HU4_GDB.gdb'
# projection = 'esri:102003'
# =========================
# Load lengths
# =========================
n_huc = 4
smooth_val = True
rivers_hucn = FM.load_data(f'{path_data}river_transect_all_huc{n_huc}.hdf5')
lengths_hucn_df = FM.load_data(f'{path_data}lengths_hw_huc{n_huc}.feather',
                               pandas_dataframe=True)
# Pick lengths larger than 100 km
lengths_hucn_df_100 = lengths_hucn_df.loc[lengths_hucn_df['length'] >= 100000]

comids = lengths_hucn_df_100['nhdplusid'].astype(str).values
ds = 50  # meters
for i_c, c in enumerate(comids):
	print(c)
	x = rivers_hucn[c]['x']
	y = rivers_hucn[c]['y']
	s = rivers_hucn[c]['s']
	if smooth_val:
		k = 3
		smooth = len(x)*1e-1
	else:
		smooth = 0
		k = 1
	x_inv = x[::-1]
	y_inv = y[::-1]
	s_inv = RF.get_reach_distances(np.array([x_inv, y_inv]).T)
	z_inv = np.zeros_like(s_inv)
	so_inv = rivers_hucn[c]['so'][::-1]
	comid_inv = rivers_hucn[c]['comid'][::-1]
	da_sqkm_inv = rivers_hucn[c]['da_sqkm'][::-1]
	w_m_inv = rivers_hucn[c]['w_m'][::-1]
	# Load data
	data_inv = {'x': x_inv, 'y': y_inv, 's': s_inv, 'z': z_inv, 'so': so_inv,
	            'comid': comid_inv, 'da_sqkm': da_sqkm_inv, 'w_m': w_m_inv}

	# Fit Complete Splines
	data_sp = RF.fit_splines_complete(
		data_inv, ds=ds, k=k, smooth=smooth, ext=0)
	s_inv = data_sp['s_poly']
	cond = s_inv <= 100000
	# Clip data
	x_inv = data_sp['x_poly'][cond]
	y_inv = data_sp['y_poly'][cond]
	z_inv = data_sp['z_poly'][cond]
	so_inv = data_sp['so_poly'][cond]
	comid_inv = data_sp['comid_poly'][cond]
	da_sqkm_inv = data_sp['da_sqkm_poly'][cond]
	w_m_inv = data_sp['w_m_poly'][cond]
	s_inv = data_sp['s_poly'][cond]

	# flip data again
	x_s = x_inv[::-1]
	y_s = y_inv[::-1]
	z_s = z_inv[::-1]
	so_s = so_inv[::-1]
	comid_s = comid_inv[::-1]
	da_sqkm_s = da_sqkm_inv[::-1]
	w_m_s = w_m_inv[::-1]
	s_s = s_inv

	# save data
	data_save = {'x': x_s, 'y': y_s, 's': s_s, 'z': z_s, 'so': so_s,
				'comid': comid_s, 'da_sqkm': da_sqkm_s, 'w_m': w_m_s}
	
	if i_c == 0:
		rivers_hucn_100 = {c: data_save}
	else:
		rivers_hucn_100[c] = data_save


# ======================================================
# Save information
# ======================================================
if smooth_val:
	FM.save_data(rivers_hucn_100, f'{path_out}',
			 file_name=f'river_transect_all_huc{n_huc}_100_smooth.hdf5')
else:
	FM.save_data(rivers_hucn_100, f'{path_out}',
				file_name=f'river_transect_all_huc{n_huc}_100.hdf5')

print('Done')


