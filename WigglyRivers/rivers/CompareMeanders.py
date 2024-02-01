# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2023-08-15
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
______________________________________________________________________________

 DESCRIPTION:
   Functions related to comparison of meander databases
______________________________________________________________________________
"""
# -----------
# Libraries
# -----------
from typing import Union, List, Tuple, Dict, Any, Optional
import time
import copy
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy import interpolate
from scipy.signal import find_peaks
from collections import Counter
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean

# Package packages
from ..utilities import general_functions as GF
from ..utilities.classExceptions import *
from ..utilities import filesManagement as FM
from . import RiverFunctions as RF


# -----------
# Functions
# -----------
def extract_closet_meanders(database_1, database_2, link_x='x_o', link_y='y_o',
                            threshold=0.8):
    """
    Description:
    ------------ 
        This function extracts the closest meander from both meander databases
        using the intersect of the linking x and y coordiantes.

        In this function we will compare meanders in database_1 with all the 
        meanders on database_2.
    ____________________________________________________________________________

    Args:
    -----------
    :param database_1: pd.Dataframe,
        Dictionary with the meander information extracted from the Rivers class.
    :param database_2: pd.Dataframe,
        Dictionary with the meander information extracted from the Rivers class.
    :param link_x: str, optional, Default: 'x_o'
        Name of the x coordinate to link the meanders.
    :param link_y: str, optional, Default: 'y_o'
        Name of the y coordinate to link the meanders.
    """
    # Prepare data to save
    data_to_save = {f'{i}_1':[] for i in database_1.columns}
    data_to_save.update({f'{i}_2':[] for i in database_2.columns})
    # Include zones for classification
    data_to_save.update({'Zone': [], 'f_oa': [], 'f_om': []})
    # Loop through all meanders
    for i_m in range(len(database_1)):
        try:
            x_o = RF.convert_str_float_list_vector(database_1[link_x].values[i_m])
        except AttributeError:
            x_o = database_1[link_x].values[i_m]

        try:
            y_o = RF.convert_str_float_list_vector(database_1[link_y].values[i_m])
        except AttributeError:
            y_o = database_1[link_y].values[i_m]
        # Save data
        for i in database_1.columns:
            data_to_save[f'{i}_1'] = [database_1[i].values[i_m]]

        # Extractr variables
        comid_o = database_1['start_comid'].values[i_m]
        curvature_side = database_1['curvature_side'].values[i_m]
        # Extract coordinates from auto that have the same comid
        sub_df = database_2[database_2['start_comid'] == comid_o]
        # Extract same curvature side
        # sub_df = sub_df[sub_df['curvature_side'] == curvature_side]
        # if len(sub_df) == 0:
        #     sub_df = database_2[database_2['comid'] == comid_o]
        #     if len(sub_df) == 0:
        #         sub_df = copy.deepcopy(database_2)

        # Extract coordinates from auto
        # Find starting and ending points close to the manual meander
        points_st_o = np.array([x_o[0], y_o[0]])
        points_end_o = np.array([x_o[-1], y_o[-1]])
        points_st_a = np.array([
            sub_df['x_start'].values, sub_df['y_start'].values]).T
        points_end_a = np.array([
            sub_df['x_end'].values, sub_df['y_end'].values]).T
        # Calculate distance
        dist_st = np.linalg.norm(points_st_a - points_st_o, axis=1)
        dist_end = np.linalg.norm(points_end_a - points_end_o, axis=1)

        i_sort_st = np.argsort(dist_st)
        i_sort_end = np.argsort(dist_end)
        # TODO: Check this calculation taking into account the inflection points

        # pick the first meanders to compare
        pick = 2
        i_compare = pd.unique(np.concatenate(
            [i_sort_st[:pick], i_sort_end[:pick]]))
        sub_df = sub_df.iloc[i_compare]
        # Find the meanders that intersect the most
        len_largest = 0
        selected_m = 0
        for i_sub in range(len(sub_df)):
            try:
                x_a = RF.convert_str_float_list_vector(
                    sub_df[link_x].values[i_sub])
            except AttributeError:
                x_a = sub_df[link_x].values[i_sub]
                
            try:
                y_a = RF.convert_str_float_list_vector(
                    sub_df[link_y].values[i_sub])
            except AttributeError:
                y_a = sub_df[link_y].values[i_sub]

            idx_int_x = np.intersect1d(x_o, x_a)
            idx_int_y = np.intersect1d(y_o, y_a)
            if len(idx_int_x) == len(idx_int_y):
                len_int = len(idx_int_x)
                if len_int > len_largest:
                    len_largest = copy.deepcopy(len_int)
                    selected_m = copy.deepcopy(i_sub)

        try:
            x_s = RF.convert_str_float_list_vector(
                sub_df[link_x].values[selected_m])
        except AttributeError:
            x_s = sub_df[link_x].values[selected_m]
        # Save the selected meander
        for i in sub_df.columns:
            data_to_save[f'{i}_2'] = [sub_df[i].values[selected_m]]
        # Perform classification
        class_value, f_oa, f_om = classify_meanders(
            x_o, x_s, threshold=threshold)
        data_to_save['Zone'] = [class_value]
        data_to_save['f_oa'] = [f_oa]
        data_to_save['f_om'] = [f_om]
        df_save = pd.DataFrame(data_to_save)
        if i_m == 0:
            database = copy.deepcopy(df_save)
        else:
            database = pd.concat([database, df_save], axis=0)
    
    database.reset_index(drop=True, inplace=True)
    return database


def classify_meanders(manual_indices, auto_indices, threshold=0.8):
    """
    Description:
    ------------
        This function performs the comparison between the manual and automatic 
        detection of meanders and classifies the comparison into four categories

        Zone I: The automatic detection is a good approximation of the manual
                detection.
        Zone II: The automatic detection is only a part of the manual detection.
        Zone III: The automatic detection is a superset of the manual detection.
        Zone IV: The automatic detection did not detect the meander.
    ___________________________________________________________________________

    Args:
    -----
    :param manual_indices: list,
        List of indices of the manual meanders.
    :param auto_indices: list,
        List of indices of the automatic meanders.
    :return: classification_value: int,
        Value of the classification.
    :return: f_oa: float,
        Fraction of the automatic meander that is inside the manual meander.
    :return: f_om: float,
        Fraction of the manual meander that is inside the automatic meander.
    """
    fst = np.intersect1d(manual_indices, auto_indices)
    f_oa = len(fst) / len(auto_indices)
    f_om = len(fst) / len(manual_indices)

    # Classification
    if f_oa >= threshold:
        if f_om >= threshold:
            # Inside (I)
            classification_value = 1
        elif f_om < threshold:
            # Underestimated (II)
            classification_value = 2
    else:
        if f_om >= threshold:
            # Overestimated (III)
            classification_value = 3
        elif f_om < threshold:
            # Outside (IV)
            classification_value = 4
    return classification_value, f_oa, f_om
