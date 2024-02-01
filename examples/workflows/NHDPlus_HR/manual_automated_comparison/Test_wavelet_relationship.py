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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Importing pyMeander Packages
from WigglyRivers import RiverDatasets
from WigglyRivers import RiverFunctions as RF
from WigglyRivers import ExtractNHDPlusHRData as NHD
from WigglyRivers.utilities import utilities as utl
from WigglyRivers.utilities import filesManagement as FM
from WigglyRivers import Logger
from WigglyRivers import WaveletTreeFunctions as WTFunc
from WigglyRivers import CompareMeanders as CM


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
path_projects = f'{os.getcwd()}/examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/manual_corrected/'
projects = utl.get_folders(path_projects)
projects = list(np.sort(projects))
path_projects_all = [f'{path_projects}{p}/' for p in projects]
path_projects_c = f'{os.getcwd()}/examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/manual_with_automatic/'
path_projects_out = [f'{path_projects_c}{p}/' for p in projects]
print(projects)
# Current projection of information
projection = 'esri:102003'


# ------------------------
# Calculate automatic
# ------------------------
bounds_array_str = 'extended'
# Select project to load
data_link = {}
for i_m, project in enumerate(projects):
    print(project)
    try:
        i_p = projects.index(project)
    except ValueError:
        raise ValueError(f'Project "{project}" not found')

    project_to_load = path_projects_out[i_p]
    # Load Tree database
    tree_df = FM.load_data(
        f'{project_to_load}/tree_scales_database.feather',
        pandas_dataframe=True)
    
    # Extract meanders
    tree_df_m = tree_df[tree_df['within_waterbody'] == 0]
    tree_df_m = tree_df_m[tree_df_m['is_meander'] == 1]

    # Load Meander database
    database = FM.load_data(
        f'{project_to_load}/meander_database_{bounds_array_str}.feather',
        pandas_dataframe=True)

    # Check common meanders between automatic and manual
    auto_df = database[database['automatic_flag'] == 1]

    # Add columns in tree_df_m
    tree_df_m['comid'] = tree_df_m['comid_c']
    tree_df_m['curvature_side'] = tree_df_m['sign_c']
    tree_df_m['x_start'] = np.array([
        tree_df_m.loc[i, 'x_inf'][0] for i in tree_df_m.index])
    tree_df_m['y_start'] = np.array([
        tree_df_m.loc[i, 'y_inf'][0] for i in tree_df_m.index])
    tree_df_m['x_end'] = np.array([
        tree_df_m.loc[i, 'x_inf'][-1] for i in tree_df_m.index])
    tree_df_m['y_end'] = np.array([
        tree_df_m.loc[i, 'y_inf'][-1] for i in tree_df_m.index])
    tree_df_m['id'] = np.array(tree_df_m.index)

    # Extract columns of interest
    # tree_df_m = tree_df_m[['x', 'y', 'wavelength_c']]
    # auto_df = auto_df[['x', 'y', 'lambda_fm', 'lambda_hm']]

    database = CM.extract_closet_meanders(
        auto_df, tree_df_m, link_x='x_inf', link_y='y_inf')

    # Test that the meanders where well picked
    # threshold = 0.75
    # sns.set(font_scale=2)
    # # Plot hexbin
    # hexplot = sns.jointplot(data=database, x='f_oa', y='f_om', kind='hex',
    #             color='#2b8cbe',
    #             xlim=(-0.05, 1.05),
    #             ylim=(-0.05, 1.05), height=10)
    # plt.axhline(y=threshold, color='k', linestyle='--')
    # plt.axvline(x=threshold, color='k', linestyle='--')
    # plt.xlabel(r'$F_{OA}=N_O/N_A$')
    # plt.ylabel(r'$F_{OM}=N_O/N_M$')
    # plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    # cbar_ax = hexplot.fig.add_axes([.82, .20, .02, .5])
    # plt.colorbar(cax=cbar_ax)
    # plt.show()

    database = pd.merge(database, tree_df_m, left_on='id_2', right_index=True,
                        suffixes=('', '_tree'))
    database = pd.merge(database, auto_df, left_on='id_1', right_on='id',
                        suffixes=('', '_auto'))

    # sub_df = tree_df_m[['x', 'y', 'wavelength_c', 'lambda_value']]
    sub_df = database[['x_1', 'y_1', 'x_2', 'y_2', 'wavelength_c',
                       'lambda_value', 'lambda_hm', 'lambda_fm']]
    # lambda_var = 'lambda_fm'
    # # Plot lambda_hm vs wavelength_c with regression line with equation
    # # Fit a line to the data
    # slope, intercept = np.polyfit(
    #     sub_df[lambda_var], sub_df['wavelength_c'], 1)
    # # Create a string for the equation
    # equation = f'y = {slope:.2f}x + {intercept:.2f}'
    # sns.set(font_scale=2)
    # sns.set_style('whitegrid')
    # sns.set_context('paper')
    # fig, ax = plt.subplots(figsize=(10, 10))
    # sns.scatterplot(data=sub_df, x=lambda_var, y='wavelength_c', ax=ax)
    # sns.regplot(data=sub_df, x=lambda_var, y='wavelength_c', ax=ax)
    # plt.text(0.1, 0.9, equation, transform=ax.transAxes)
    # plt.gca().set_xscale('log')
    # plt.gca().set_yscale('log')
    # plt.xlabel(r'$\lambda_{HM}$')
    # plt.ylabel(r'$\lambda_{CWT}$')
    # plt.show()

    if i_m == 0:
        database_all = sub_df.copy()
    else:
        database_all = pd.concat([database_all, sub_df], ignore_index=True)

for lambda_var in ['lambda_fm', 'lambda_hm']:
    slope, intercept = np.polyfit(
        database_all[lambda_var], database_all['wavelength_c'], 1)
    # TODO: Try with a loggaritmic fitting 
    equation = f'y = {slope:.2f}x + {intercept:.2f}'
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    sns.set_context('paper')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=database_all, x=lambda_var, y='wavelength_c', ax=ax)
    sns.regplot(data=database_all, x=lambda_var, y='wavelength_c', ax=ax)
    plt.text(0.1, 0.9, equation, transform=ax.transAxes)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel(lambda_var)
    plt.ylabel(r'$\lambda$')
    plt.savefig(f'examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/{lambda_var}_vs_lambda.png')
    plt.close('all')

FM.save_data(database_all, path_output=f'examples/workflows/NHDPlus_HR/manual_automated_comparison/meander_comparison/characterization/',
                file_name=f'compare_wavelength_vs_lambda.csv')

