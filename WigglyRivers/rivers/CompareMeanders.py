# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2025-02-16
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
Functions related to comparison of meander databases
"""
# -----------
# Libraries
# -----------
import copy
import numpy as np
import pandas as pd

# Package packages
from . import RiverFunctions as RF


# -----------
# Functions
# -----------
def extract_closet_meanders(
    database_1: pd.DataFrame,
    database_2: pd.DataFrame,
    link_x: str = "x_o",
    link_y: str = "y_o",
    threshold: float = 0.8,
) -> pd.DataFrame:
    """Extracts the closest meander from both meander databases
    using the intersect of the linking x and y coordiantes.

    In this function we will compare meanders in database_1 with all the
    meanders on database_2.

    Args:
        database_1 (pd.Dataframe): Dataframe with the meander information
            extracted from the Rivers class to compare with database 2.
        database_2 (pd.Dataframe): Dataframe with the meander information
            extracted from the River Class to compare with database 1.
        link_x (str, optional): linking variable for x coordiantes.
            Defaults to "x_o".
        link_y (str, optional): linking variable for y coordiantes.
            Defaults to "y_o".
        threshold (float, optional): Threshold for classification among zones.
            Defaults to 0.8.

    Returns:
        pd.Dataframe: Dataframe with classification done among the meanders.
    """
    # Prepare data to save
    data_to_save = {f"{i}_1": [] for i in database_1.columns}
    data_to_save.update({f"{i}_2": [] for i in database_2.columns})
    # Include zones for classification
    data_to_save.update({"Zone": [], "f_oa": [], "f_om": []})
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
            data_to_save[f"{i}_1"] = [database_1[i].values[i_m]]

        # Extractr variables
        comid_o = database_1["start_comid"].values[i_m]
        # Extract coordinates from auto that have the same comid
        sub_df = database_2[database_2["start_comid"] == comid_o]
        # Extract same curvature side

        # Extract coordinates from auto
        #  Find starting and ending points close to the manual meander
        points_st_o = np.array([x_o[0], y_o[0]])
        points_end_o = np.array([x_o[-1], y_o[-1]])
        points_st_a = np.array([sub_df["x_start"].values, sub_df["y_start"].values]).T
        points_end_a = np.array([sub_df["x_end"].values, sub_df["y_end"].values]).T
        # Calculate distance
        dist_st = np.linalg.norm(points_st_a - points_st_o, axis=1)
        dist_end = np.linalg.norm(points_end_a - points_end_o, axis=1)

        i_sort_st = np.argsort(dist_st)
        i_sort_end = np.argsort(dist_end)

        # Pick the first meanders to compare
        pick = 2
        i_compare = pd.unique(np.concatenate([i_sort_st[:pick], i_sort_end[:pick]]))
        sub_df = sub_df.iloc[i_compare]
        # Find the meanders that intersect the most
        len_largest = 0
        selected_m = 0
        for i_sub in range(len(sub_df)):
            try:
                x_a = RF.convert_str_float_list_vector(sub_df[link_x].values[i_sub])
            except AttributeError:
                x_a = sub_df[link_x].values[i_sub]

            try:
                y_a = RF.convert_str_float_list_vector(sub_df[link_y].values[i_sub])
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
            x_s = RF.convert_str_float_list_vector(sub_df[link_x].values[selected_m])
        except AttributeError:
            x_s = sub_df[link_x].values[selected_m]
        # Save the selected meander
        for i in sub_df.columns:
            data_to_save[f"{i}_2"] = [sub_df[i].values[selected_m]]
        # Perform classification
        class_value, f_oa, f_om = classify_meanders(x_o, x_s, threshold=threshold)
        data_to_save["Zone"] = [class_value]
        data_to_save["f_oa"] = [f_oa]
        data_to_save["f_om"] = [f_om]
        df_save = pd.DataFrame(data_to_save)
        if i_m == 0:
            database = copy.deepcopy(df_save)
        else:
            database = pd.concat([database, df_save], axis=0)

    database.reset_index(drop=True, inplace=True)
    return database


def classify_meanders(
    manual_indices: list, auto_indices: list, threshold: float = 0.8
) -> tuple:
    """Comparison between the manual and automatic
    detection of meanders and classifies the comparison into four categories

    Zone I: The automatic detection is a good approximation of the manual detection.

    Zone II: The automatic detection is only a part of the manual detection.

    Zone III: The automatic detection is a superset of the manual detection.

    Zone IV: The automatic detection did not detect the

    Args:
        manual_indices (list): List of indices of the manually selected meanders.
        auto_indices (list): List of indices of the automatically selected meanders.
        threshold (float, optional): Threshold for classification among zones.
            Defaults to 0.8.

    Returns:
        tuple: classification_value, f_oa, f_om
            classification_value (int): Value of the classification.
            f_oa (float): Fraction of the automatic meander that is inside
                the manual meander.
            f_om (float): Fraction of the manual meander that is inside
                the automatic meander.
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
