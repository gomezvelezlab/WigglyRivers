
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
path_clip = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/WBD/cliped_data/'
path_cwt = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/WBD/cwt_data/'
# =========================
# Load lengths
# =========================
n_huc = 8
utl.cr_folder(f'{path_cwt}HUC{n_huc}/Figures/')
smooth_val = True
if smooth_val:
    rivers_hucn = FM.load_data(
        f'{path_clip}river_transect_all_huc{n_huc}_100_smooth.hdf5')
else:
    rivers_hucn = FM.load_data(
        f'{path_clip}river_transect_all_huc{n_huc}_100.hdf5')

comids = list(rivers_hucn.keys())
rivers = Rivers()
for i_c, c in enumerate(comids):
    print(c)
    x = rivers_hucn[c]['x']
    y = rivers_hucn[c]['y']
    s = rivers_hucn[c]['s']
    so = rivers_hucn[c]['so']
    comid = rivers_hucn[c]['comid']
    da_sqkm = rivers_hucn[c]['da_sqkm']
    w_m = rivers_hucn[c]['w_m']
    z = np.zeros_like(s)*np.nan

    # Add river
    rivers.add_river(c, x, y, s, z, so, comid, da_sqkm,
                     w_m, resample_flag=False)
    # Calculate curvature
    rivers[c].calculate_curvature()
    # Calculate CWT
    rivers[c].get_cwt_curvature(mother='MORLET')
    rivers[c].get_cwt_angle(mother='MORLET')

    # if i_c <= 20:
    #     graphs.plot_river_spectrum_compiled(
    #         rivers[c], only_significant=True)
    #     plt.savefig(f'{path_cwt}HUC{n_huc}/Figures/{i_c}_{c}_cwt.png', dpi=300)
    #     plt.close('all')
    # else:
    #     aaa
    if i_c % 900 == 0 and i_c > 0:
        print('Saving and Reseting rivers')
        if smooth_val:
            cwt_data = rivers.extract_cwt_data()
            variables = ['x', 'y', 's', 'c', 'angle', 'comid', 'wavelength_c',
                         'power_c_sig', 'power_angle_sig']
            cwt_data_save = {i: {j: cwt_data[i][j] for j in variables}
                             for i in list(cwt_data.keys())}
            FM.save_data(cwt_data_save, path_output=f'{path_cwt}HUC{n_huc}/',
                         file_name=f'cwt_data_huc{n_huc}_smooth_{i_c}.hdf5')
            rivers.save_cwt_data(
                path_output=f'{path_cwt}HUC{n_huc}/',
                file_name=f'cwt_data_huc{n_huc}_smooth_{i_c}.hdf5')
        else:
            rivers.save_cwt_data(path_output=f'{path_cwt}HUC{n_huc}/',
                                file_name=f'cwt_data_huc{n_huc}_{i_c}.hdf5')
        rivers = Rivers()

    
# ======================================================
# Save information
# ======================================================
cwt_data = rivers.extract_cwt_data()
variables = ['x', 'y', 's', 'c', 'angle', 'comid', 'wavelength_c',
            'power_c_sig', 'power_angle_sig']
cwt_data_save = {i: {j: cwt_data[i][j] for j in variables}
                    for i in list(cwt_data.keys())}
FM.save_data(cwt_data_save, path_output=f'{path_cwt}HUC{n_huc}/',
                file_name=f'cwt_data_huc{n_huc}_smooth_{i_c}.hdf5')
# if smooth_val:
#     # rivers.save_rivers(
#     #     path_output=f'{path_cwt}',
#     #     file_name=f'river_transect_all_huc{n_huc}_cwt_smooth.hdf5')
#     rivers.save_cwt_data(path_output=f'{path_cwt}HUC{n_huc}/',
#                          file_name=f'cwt_data_huc{n_huc}_smooth_{i_c}.hdf5')
# else:
#     # rivers.save_rivers(
#     #     path_output=f'{path_cwt}',
#     #     file_name=f'river_transect_all_huc{n_huc}_cwt.hdf5')
#     rivers.save_cwt_data(path_output=f'{path_cwt}HUC{n_huc}/',
#                          file_name=f'cwt_data_huc{n_huc}_{i_c}.hdf5')

print('Done')


