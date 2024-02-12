
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
# huc_id = '0602'
# huc_id = '0104'
# huc_id = '0601'
# huc_id = '0513'
path_wbd = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/WBD/raw_data/'
path_output = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/WBD/processed_data/'
path_nhd = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/raw_data/'
path_nhd_output = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/process_data/'
# path_river_routing = f'{path_output}/river_routing/'
# path_meander_info = f'{path_output}/meanders/'
# file_name = f'NHDPlus_H_{huc_id}_HU4_GDB.gdb'
projection = 'esri:102003'
# =========================
# Load WBD
# =========================
wbd_gdb = f'{path_wbd}/WBD_National_GDB/WBD_National_GDB.gdb'
wbd_gdb = f'{path_wbd}/WBD_06_HU2_GDB/WBD_06_HU2_GDB.gdb'

# Load geodatabase
print('Loading WBD...')
wbd_shp_12 = FM.read_gbd(wbd_gdb, layer='WBDHU12')
to_huc12 = wbd_shp_12['tohuc'].values
# =========================
# Extract Headwaters HUCN
# =========================
# ----------------------------
# Extract HUC 4 values 
# ----------------------------
huc_4 = np.array([i[:4] for i in wbd_shp_12['huc12']])
wbd_shp_12['huc04'] = huc_4
# ----------------------------
# Extract HUC N values 
# ----------------------------
n_huc = 8
print(f'Extracting HUC{n_huc} Headwaters...')
huc_n = np.array([i[:n_huc] for i in wbd_shp_12['huc12']])
# Extract unique values at that huc level
wbd_shp_12[f'huc{n_huc}'] = huc_n
huc_n_unique = np.unique(huc_n)
hw_huc_n = list(huc_n_unique)
# Iterlate over the unique values
for h_n in huc_n_unique:
    print(h_n)
    # Extract the subset of the WBD that has the same huc{n_huc} value
    subset_wbd_n = wbd_shp_12[wbd_shp_12[f'huc{n_huc}'] == h_n]
    # Extract the subset of the WBD that has a different huc{n_huc} value
    subset_wbd_not_n = wbd_shp_12[wbd_shp_12[f'huc{n_huc}'] != h_n]
    subset_huc_12 = subset_wbd_n['huc12'].values
    subset_to_huc_12 = subset_wbd_not_n['tohuc'].values
    # Loop over the huc12 values to check:
    #  if there is at least one 'huc12' that is outside the 'huc{n_huc}' subset
    #  in the 'tohuc' columnm, then huc{n_huc} is not a headwater
    for h_12 in subset_huc_12:
        # Check if there is any huc12 draining from a different hucn
        len_huc = len(subset_to_huc_12[subset_to_huc_12 == h_12])
        if len_huc > 0:
            print(len_huc)
            try:
                index_n = hw_huc_n.index(h_n)
            except ValueError:
               break 
            hw_huc_n.pop(index_n)
            break 
print(hw_huc_n)
# ======================================================
# Disolve individual polygons at the huc{n_huc} level
# ======================================================
# Set huc{n_huc} as index
wbd_shp_12.set_index(f'huc{n_huc}', inplace=True)
# Select only the headwaters
wbd_shp_huc_n = wbd_shp_12.loc[hw_huc_n]
# Reset index
wbd_shp_12.reset_index(inplace=True)
wbd_shp_huc_n.reset_index(inplace=True)
# Dissolve the polygons
wbd_shp_huc_n_dissolved = wbd_shp_huc_n.dissolve(by=f'huc{n_huc}')
# Save the headwaters
path_wbd_out = f'{path_output}/WBD_headwaters/HUC{n_huc}/'
file_name = wbd_gdb.split('/')[-1].split('.')[0]
# remove 'loaddate' from dataframe
wbd_shp_huc_n_dissolved = wbd_shp_huc_n_dissolved.drop(columns=['loaddate'])
FM.save_data(wbd_shp_huc_n_dissolved, path_wbd_out,
             f'{file_name}_HUC{n_huc}.shp')

# ======================================================
# Extract Rivers
# ======================================================
# Value for huc 8
n_huc = 8
f_codes_to_remove = [
    56600,  # Coastline
    # 33400,  # Connector
    # 33600,  # Canal/Ditch
    # 33601,  # Canal/Ditch
    # 33603,  # Canal/Ditch
    42800,  # Pipeline
    42801,  # Pipeline
    42802,  # Pipeline
    42803,  # Pipeline
    42804,  # Pipeline
    42805,  # Pipeline
    42806,  # Pipeline
    42807,  # Pipeline
    42808,  # Pipeline
    42809,  # Pipeline
    42810,  # Pipeline
    42811,  # Pipeline
    42812,  # Pipeline
    42813,  # Pipeline
    42814,  # Pipeline
    42815,  # Pipeline
    42816,  # Pipeline
]
# -------------------------
# Open WBD Headwaters
# -------------------------
wbd_shp_huc_n_hw = FM.load_data(f'{path_wbd_out}/{file_name}_HUC{n_huc}.shp')
huc_4 = wbd_shp_huc_n_hw['huc04'].values
huc_4_unique = np.unique(huc_4)
# -------------------------
# Extract Rivers
# -------------------------
for i_huc, huc_4_val in enumerate(huc_4_unique):
    if i_huc < 1:
       continue 
    print(f'Extracting Rivers for HUC4 {huc_4_val}...')
    shp_file = f'{path_nhd}NHDPlus_H_{huc_4_val}_HU4_GDB/NHDPlus_H_{huc_4_val}_HU4_GDB.gdb'
    river_shp = FM.read_gbd(shp_file, layer='NHDFlowline')
    comid_label = 'NHDPlusID'
    river_shp[comid_label] = river_shp[comid_label].astype('int64')
    river_shp[comid_label] = river_shp[comid_label].astype(str)
    river_shp.set_index(comid_label, inplace=True)
    # Add tables
    tables_nhd = ['NHDPlusFlowlineVAA', 'NHDPlusEROMMA']
    for t in tables_nhd:
        table = FM.read_gbd(shp_file, layer=t)
        table[comid_label] = table[comid_label].astype('int64')
        table[comid_label] = table[comid_label].astype(str)
        table.set_index(comid_label, inplace=True)
        cols_to_use = table.columns.difference(river_shp.columns)
        river_shp = pd.merge(river_shp, table[cols_to_use], left_index=True,
                            right_index=True, how='left')
    # -------------------------
    # join withn WBD
    # -------------------------
    subset_huc_4 = wbd_shp_huc_n_hw[wbd_shp_huc_n_hw[f'huc04'] == huc_4_val]
    huc_n = subset_huc_4[f'huc{n_huc}'].values
    huc_n_unique = np.unique(huc_n)
    huc_n_reach_code = np.array([i[:n_huc] for i in river_shp['ReachCode']])
    river_shp[f'ReachCode_huc{n_huc}'] = huc_n_reach_code
    for i_h, h_n in enumerate(huc_n_unique):
        subset_huc_n = subset_huc_4[subset_huc_4[f'huc{n_huc}'] == h_n]
        rivers_hw = gpd.sjoin(river_shp, subset_huc_n, how='inner',
                              predicate='within')
        # rivers_hw_reachcode = river_shp[river_shp[
        #     f'ReachCode_huc{n_huc}'] == h_n]
        # -------------------------
        # Extract Mainstem 
        # -------------------------
        rivers_hw = rivers_hw[
            rivers_hw['StreamLeve'] == rivers_hw['StreamLeve'].min()]
        if i_h == 0:
            rivers_hw_all = rivers_hw
        else:
            rivers_hw_all = pd.concat([rivers_hw_all, rivers_hw])

    # change columns to lower case
    rivers_hw_all.reset_index(inplace=True) 
    rivers_hw_all = rivers_hw_all.rename(columns=lambda x: x.lower())
    rivers_hw_all['tonode'] = rivers_hw_all['tonode'].astype('int64')
    rivers_hw_all['tonode'] = rivers_hw_all['tonode'].astype(str)
    rivers_hw_all['fromnode'] = rivers_hw_all['fromnode'].astype('int64')
    rivers_hw_all['fromnode'] = rivers_hw_all['fromnode'].astype(str)
    # -------------------------
    # Extract coordinates
    # -------------------------
    path_nhd_out = f'{path_nhd_output}NHDPlus_H_{huc_4_val}_HU4_GDB/HW_HUC_{n_huc}/'
    nhd_obj = NHD(path_nhd_out)
    # Project database
    rivers_hw_all_projected = rivers_hw_all.to_crs(projection)
    # Extract coordinates
    coords_all = {
        comid_label.lower(): rivers_hw_all[comid_label.lower()].values,
        'ftype': rivers_hw_all['ftype'].values,
    }
    coords_all_projected = {
        comid_label.lower(): rivers_hw_all[comid_label.lower()].values,
        'ftype': rivers_hw_all['ftype'].values,
    }
    coords = np.zeros((rivers_hw_all.shape[0], 6)) * np.nan
    coords_projected = np.zeros((rivers_hw_all_projected.shape[0], 6)) * np.nan
    for i in range(rivers_hw_all.shape[0]):
        try:
            bounds = rivers_hw_all.geometry[i].geoms[0]
        except TypeError:
            bounds = rivers_hw_all.geometry[i][0]
        coords[i], coords_all_comid = nhd_obj._get_coordinates(
            bounds, rivers_hw_all[comid_label.lower()].values[i])
        coords_all.update(coords_all_comid)
        if projection is not None:
            # Extract coordinates in projected coordinates
            try:
                bounds_proj = rivers_hw_all_projected.geometry[i].geoms[0]
            except TypeError:
                bounds_proj = rivers_hw_all_projected.geometry[i][0]
            coords_projected[i], coords_all_comid_projected = \
                nhd_obj._get_coordinates(
                    bounds_proj,
                    rivers_hw_all_projected[comid_label.lower()].values[i])
            coords_all_projected.update(coords_all_comid_projected)
    FM.save_data(coords_all, path_nhd_out,
                 f'coords_all_huc{n_huc}.hdf5')
    FM.save_data(coords_all_projected, path_nhd_out,
                 f'coords_all_huc{n_huc}_projected.hdf5')
    # -------------------------
    # Link Network
    # -------------------------
    # Extract headwaters
    hw_comid = rivers_hw_all[rivers_hw_all['startflag'] == 1][
        comid_label.lower()].values
    reach_generator = RE.CompleteReachExtraction(rivers_hw_all)
    reach_generator.coords_all = coords_all_projected
    reach_generator.pre_loaded_coords = True
    comid_network_all = {comid: [] for comid in hw_comid}
    coords_mapped = {comid: [] for comid in hw_comid}
    length_reach = []
    da = []
    for comid in hw_comid:
        # Map network
        comid_network, _ = reach_generator.map_complete_reach(
            comid, do_not_overlap=False)
        comid_network_all[comid] = comid_network
        # Extract coordinates
        coords_data = reach_generator.map_coordinates(
            comid_network, f'{path_nhd_out}/coords_all_huc{n_huc}_projected.hdf5')
        coords_data.reset_index(inplace=True)
        coords_data = {k: coords_data[k].values for k in coords_data.columns}
        coords_mapped[comid] = coords_data
        length_reach.append(coords_data['s'][-1])
        da.append(coords_data['da_sqkm'][-1])
    coords_mapped['comids'] = comid_network_all

    rivers_hw_all.set_index(comid_label.lower(), inplace=True)
    hw_river_only = rivers_hw_all.loc[hw_comid]
    lengths_df = pd.DataFrame({comid_label.lower(): hw_comid,
                               'length': length_reach,
                               'da_sqkm': da,
                               'huc04': hw_river_only['huc04'].values,
                               f'huc{n_huc}': hw_river_only[f'huc{n_huc}'].values,
                               })

    # path_nhd_output = f'examples/workflows/NHDPlus_HR/NHDPlus_HR/process_data/'
    path_nhd_out = f'{path_nhd_output}NHDPlus_H_{huc_4_val}_HU4_GDB/HW_HUC_{n_huc}/'
    
    FM.save_data(comid_network_all, path_nhd_out,
                 f'comid_network_all_huc{n_huc}.hdf5')
    FM.save_data(coords_mapped, path_nhd_out,
                 f'river_transect_all_huc{n_huc}.hdf5')
    FM.save_data(lengths_df, path_nhd_out,
                 f'lengths_hw_huc{n_huc}.csv')
    rivers_hw_all.reset_index(inplace=True)
    rivers_hw_all = rivers_hw_all.drop(columns=['fdate'])
    rivers_hw_all_projected = rivers_hw_all_projected.drop(columns=['fdate'])
    # Remove duplicate names
    rivers_hw_all = rivers_hw_all.loc[:, ~rivers_hw_all.columns.duplicated()]
    rivers_hw_all_projected = rivers_hw_all_projected.loc[:, ~rivers_hw_all_projected.columns.duplicated()]
    FM.save_data(rivers_hw_all, path_nhd_out,
                 f'hw_mainstem_huc{n_huc}.shp')
    FM.save_data(rivers_hw_all_projected, path_nhd_out,
                 f'hw_mainstem_huc{n_huc}_projected.shp')
    


# ======================================================
# River Routing
# ======================================================

# ======================================================
# Plot results
# ======================================================
# plot the headwaters
n_huc = 8
fig, axs = plt.subplots(2, 1, figsize=(10, 10))
ax = axs[0]
wbd_shp_12.plot(ax=ax, column=f'huc{n_huc}', alpha=0.5, cmap='Set3')
wbd_shp_huc_n_dissolved.plot(ax=ax, color='skyblue', edgecolor='black', alpha=1)
# plot the rivers
for i_huc, huc_4_val in enumerate(huc_4_unique):
    rivers_hw_all = FM.load_data(
        f'{path_nhd_output}NHDPlus_H_{huc_4_val}_HU4_GDB/HW_HUC_{n_huc}/hw_mainstem_huc{n_huc}.shp')
    rivers_hw_all.plot(ax=ax, alpha=1, color='b', linewidth=1)
    lengths_df = FM.load_data(
        f'{path_nhd_output}NHDPlus_H_{huc_4_val}_HU4_GDB/HW_HUC_{n_huc}/lengths_hw_huc{n_huc}.csv',
        pandas_dataframe=True)
    if i_huc == 0:
        lengths_df_all = lengths_df
    else:
        lengths_df_all = pd.concat([lengths_df_all, lengths_df])

# Remove axis
ax.set_axis_off()
ax.set_title(f'HUC{n_huc} Headwaters')

# Plot histogram of lengths
ax = axs[1]
sns.histplot(lengths_df_all['length']/1000, ax=ax, color='skyblue')
ax.set_xlabel('Length (km)')

# rivers_hw_all.loc[comid_network].plot(ax=ax, alpha=1, color='b', linewidth=1)
# rivers_hw_reachcode.plot(ax=ax, alpha=0.5, column='StreamLeve', cmap='YlOrRd') 
plt.savefig(f'test_huc_06.png')
plt.show()

print('done')
