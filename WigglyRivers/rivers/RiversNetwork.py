# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2021-04-18
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
______________________________________________________________________________

 DESCRIPTION:
   Scripts related to meander creation and fitting.
______________________________________________________________________________
"""
# -----------
# Libraries
# -----------
# System
import time
import copy
from typing import Union, List
import inspect
import uuid
from joblib import Parallel, delayed

# Data Management
import numpy as np

# Interpolation
import pandas as pd
from scipy import stats as st
from scipy.signal import find_peaks
import logging

# Plots
import matplotlib.pyplot as plt

# Package packages
from .. import VersionControl as VC
from .ReachExtraction import CompleteReachExtraction as CRE
from . import RiverFunctions as RF
from ..utilities import filesManagement as FM
from ..wavelet_tree import WaveletTreeFunctions as WTFunc
from ..wavelet_tree import waveletFunctions as cwt_func
from ..utilities.classExceptions import *
from ..utilities import graphs
from ..utilities.Logger import Logger
from ..wavelet_tree.TreeScale import RiverTreeScales
from ..utilities import interactivePlotly as iplot


# ---------------------
# Logging
# ---------------------
logging.basicConfig(handlers=[logging.NullHandler()])

# -----------
# parameters
# -----------
CALC_VARS = [
    "x_inf",
    "y_inf",
    "s_inf",
    "x_c_max",
    "y_c_max",
    "c_max",
    "s_c_max",
    "lambda_fm",
    "lambda_hm",
    "L_fm",
    "L_hm",
    "sigma_fm",
    "sigma_hm",
    "so",
    "skewness",
    "flatness",
    "R_hm",
    "curvature_side",
    "a_fm",
    "lambda_fm,u",
    "lambda_fm,d",
    "a_hm",
    "lambda_hm,u",
    "lambda_hm,d",
    "A_hm",
    "A_fm",
    "lambda",
    "FF",
    "L_l",
    "L_n",
    "s_n",
    "s_l",
]


# -----------
# Classes
# -----------
class RiverDatasets:
    """
    This class generates rivers from NHD Information.

    ===================== =====================================================
    Attribute             Description
    ===================== =====================================================
    scale                 Scaling the river variables.
    translate             Translate river variables
    rivers                dictionary with rivers
    ===================== =====================================================

    The following are the methods of the class.

    =============================== =====================================================
    Methods                         Description
    =============================== =====================================================
    add_river                       Add river network
    add_files                       Add path to NHD files and HUC 04 of the network.
                                    These files have to be created beforehand with the
                                    pyDatAPros package.
    map_network                     Map the complete network from a list of start comids.
    load_linking_network            Load the linking network from a file.
    load_extracted_in_comid_network Load the comid extracted in order to map
                                    the complete network.
    get_reaches_from_network        Get reach coordinates from the comid network
                                    dictionary.
    get_reach                       Extract coordinates of a complete reach from the NHD
                                    dataset.
    get_metric_database             Extract the meander metrics.
    save_databases_meanders         Save meander metric database.
    save_rivers                     Save river files.
    load_rivers_network             Load rivers dataset.
    plot_rivers                     Plot river network.
    =============================== =====================================================
    """

    def __init__(self, scale=1, translate=False, logger=None):
        # -------------------
        # Attributes
        # -------------------
        if logger is None:
            self._logging = logging.getLogger(self.__class__.__name__)
        else:
            self._logging = logger
            self._logging.info(f"Start Logger in {self.__class__.__name__}")

        self.path_data = ""
        self.huc04 = ""
        self.info_file = ""
        self.coords_file = ""
        self.data_info_df = None
        self.scale = scale
        self.translate = translate
        self.id_values = []
        self.rivers = {
            "version": VC.VERSION_FILES,
            "id_values": self.id_values,
            "scale": self.scale,
            "translate": self.translate,
        }
        self.reach_generator = None
        # Cleaning of NHD data
        self.f_codes_to_remove = [
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
        return

    # --------------------
    # Getters
    # --------------------
    @property
    def logger(self):
        return self._logging

    # --------------------
    # Core Methods
    # --------------------
    def __getitem__(self, item):
        return self.rivers[item]

    def add_river(
        self,
        id_value,
        x,
        y,
        s=None,
        z=None,
        so=None,
        comid=None,
        da_sqkm=None,
        w_m=None,
        huc04=np.nan,
        huc_n=np.nan,
        start_comid=np.nan,
        resample_flag=True,
        within_waterbody=None,
        uid=None,
        kwargs_resample=None,
        scale_by_width=False,
    ):
        """
        Description:
        -------------
            Add river into the class
        ________________________________________________________________________

        Args:
        ------------
        :param id_value: int,
            ID of the river.
        :param x: np.array,
            X coordinates of the river.
        :param y: np.array,
            Y coordinates of the river.
        :param s: np.array,
            Distance long the river.
        :param z: np.array,
            Elevation of the river
        :param so: np.array, optional,
            Stream order of each coordinate of the river
        :param comid: np.array, optional,
            Comid of each coordinate of the river
        :param da: np.array, optional,
            Drainage area of each coordinate of the river
        :param w: np.array, optional,
            Width of the river
        :param huc04: str, optional,
            HUC 04 of the river
        :param huc_n: str, optional,
            level of the HUC of the river
        :param start_comid: int, optional,
            Comid of the start of the river
        :param resample_flag: bool, optional,
            If the splines have been calculated
        :param uid: str, optional,
            Unique ID of the river
        :param method_spline: str, optional, default='geometric_mean'
            Method used to calculate the distance between points in the spline
            options are: 'min', 'mean', and 'geometric_mean'
        """
        try:
            self.id_values.index(id_value)
            self.logger.warning(f"ID {id_value} already exists. Ovewriting")
        except ValueError:
            self.id_values.append(id_value)
            pass
        self.rivers.update({"id_values": self.id_values})
        if uid is None:
            uid = str(uuid.uuid1())
        self.rivers[id_value] = RiverTransect(
            uid,
            id_value,
            x,
            y,
            s,
            z,
            so,
            comid,
            da_sqkm,
            w_m,
            huc04,
            huc_n,
            start_comid,
            resample_flag=resample_flag,
            within_waterbody=within_waterbody,
            kwargs_resample=kwargs_resample,
            scale_by_width=scale_by_width,
            logger=self.logger,
        )
        if self.translate:
            self.rivers[id_value].translate_xy_start()
        self.rivers[id_value].scale_coordinates(self.scale)
        return

    def add_files(
        self,
        path_data: str,
        huc04: str,
        path_coords: Union[str, None] = None,
        comid_id: str = "nhdplusid",
        load_coords: bool = False,
    ) -> None:
        """
        Description:
        ------------

            This method adds the path to the NHD files and links the coordinates
            of the river network for reach extraction. It also creates the
            reach_generator object.
        ________________________________________________________________________

        Args:
        -----------
        :param path_data: str
            Path to the NHD files.
        :type path_data: str
        :param huc04: str
            HUC04 of the river network.
        :type huc04: str
        :param path_coords: str, default=None
            Path to the coordinates of the river network.
        :type path_coords: str
        :param comid_id: str, default='nhdplus_id'
            Name of the column with the comid id.
        :type comid_id: str
        :return: None
        """

        if path_coords is None:
            path_coords = path_data
        self.path_data = path_data
        self.huc04 = huc04
        # self.info_file = f'{path_data}table_HUC04_{huc04}.csv'
        if path_data.split(".")[-1] not in ("csv", "feather"):
            self.info_file = f"{path_data}file_tables_raw.feather"
        else:
            self.info_file = f"{path_data}"
        # self.coords_file = f'{path_data}HUC04_{huc04}_coordinates_p_102003.hdf5'
        if path_coords.split(".")[-1] not in ("hdf5", "h5"):
            self.coords_file = f"{path_coords}file_coords_p_102003_raw.hdf5"
        else:
            self.coords_file = f"{path_coords}"
        try:
            self.data_info_df = FM.load_data(
                self.info_file,
                pandas_dataframe=True,
                converters={
                    "HUC08": str,
                    "HUC10": str,
                    "HUC04": str,
                    "HUC06": str,
                    "HUC12": str,
                },
            )
        except TypeError:
            # Adding HUCs to the dataframe
            self.data_info_df = FM.load_data(
                self.info_file, pandas_dataframe=True
            )
            data_huc4 = copy.deepcopy(self.data_info_df)
            huc_02 = [i[:2] for i in data_huc4["ReachCode"]]
            huc_04 = [i[:4] for i in data_huc4["ReachCode"]]
            huc_06 = [i[:6] for i in data_huc4["ReachCode"]]
            huc_08 = [i[:8] for i in data_huc4["ReachCode"]]
            huc_10 = [i[:10] for i in data_huc4["ReachCode"]]
            # Add values to the dataframe
            data_huc4.loc[:, "HUC02"] = huc_02
            data_huc4.loc[:, "HUC04"] = huc_04
            data_huc4.loc[:, "HUC06"] = huc_06
            data_huc4.loc[:, "HUC08"] = huc_08
            data_huc4.loc[:, "HUC10"] = huc_10
            self.data_info_df = data_huc4

        # Cheange headers to lower case
        self.data_info_df.columns = [
            i.lower() for i in self.data_info_df.columns
        ]
        # Clean dataset
        self._clean_nhd_data()
        # Create reach generator
        self.reach_generator = CRE(
            self.data_info_df, comid_id=comid_id, logger=self.logger
        )
        # Load Coordinates
        if load_coords:
            self.reach_generator.load_coords(self.coords_file)
        return

    def _clean_nhd_data(self):
        # Pick small divergence
        self.data_info_df = self.data_info_df[
            self.data_info_df["divergence"] <= 1
        ]
        # Remove FCodes that are not streams
        try:
            for fcode_remove in self.f_codes_to_remove:
                self.data_info_df = self.data_info_df[
                    self.data_info_df["fcode"] != fcode_remove
                ]
        except KeyError:
            self.logger.warning("No fcode column in the dataframe")
        # Delete inconsistencies with stream order and the calculated stream
        #   order
        try:
            self.data_info_df = self.data_info_df[
                self.data_info_df["streamorde"]
                == self.data_info_df["streamcalc"]
            ]
        except KeyError:
            print("No stream order calc in the dataframe")
        # Delete isolated flowlines (comids where startflag and terminalflag
        #   are equal to 1)
        try:
            self.data_info_df = self.data_info_df[
                ~(
                    (self.data_info_df["startflag"] == 1)
                    & (self.data_info_df["terminalfl"] == 1)
                )
            ]
        except KeyError:
            print("No startflag and terminalflag in the dataframe")
        return

    def map_network(
        self,
        method: str = "upstream",
        start_comids: Union[float, str, list, np.ndarray, None] = None,
        huc: int = 4,
        cut_comid_number: int = 3,
        max_num_comids: int = None,
        path_out: str = None,
        extension: str = "feather",
    ) -> None:
        """
        Description:
        -------------
            Map the complete network from a list of start comids. Two lists
            will be created within the reach_generator object. One with the
            complete network as a dictionary and the other with the linking
            between the comids as a pandas dataframe.

            If a path_out is provided, the complete network will saved as an
            'hdf5' file with the string headwaters COMIDS as keys. The
            dataframe will be saved as with the extension given. By default
            it will be saved as 'feather'.

            There are two methods to map the network. The first one is
            'upstream' and the second one is 'downstream'. The first one
            starts from the terminal nodes and goes upstream recursively
            mapping the network. The second one starts from the headwaters and
            goes downstream until it reaches the terminal nodes or nodes
            that have already been extracted. The default method is 'upstream'
            because it works faster and it is more efficient.
        ________________________________________________________________________

        Args:
        -----
        :param method: str, default='upstream'
            Method to map the network. Options are 'upstream' and 'downstream'
        :type methods: str
        :params start_comids: list,
            List of start comids
        :type start_comids: list, np.ndarray
        :params huc: int,
            HUC number to cut to in the network
        :type huc: int
        :params cut_comid_number: int,
            Minimum number of comids per complete reach extraction
        :type cut_comid_number: int
        :param max_num_comids: int,
            Maximum number of comids per complete reach extraction
        :type max_num_comids: int
        :param path_out: str,
            Path to save the complete network
        :type path_out: str
        :param extension: str,
            Extension of the file to save the complete network
        :type extension: str
        :return: None
        """
        self.logger.info("Mapping the complete network")

        if method == "downstream":
            self.logger.info("Mapping through downstream method")
            self.reach_generator.map_complete_network(
                start_comids=start_comids,
                huc_number=huc,
                cut_comid_number=cut_comid_number,
                max_num_comids=max_num_comids,
                do_not_overlap=True,
            )
        elif method == "upstream":
            self.logger.info("Mapping through upstream method")
            self.reach_generator.map_complete_network_down_up(huc_number=huc)
        else:
            self.logger.error(
                f"Methods {method} not available. Please, use "
                'either "upstream" or "downstream"'
            )
            raise ValueError(
                f"Methods {method} not available. Please, use "
                'either "upstream" or "downstream"'
            )

        # Save the complete network
        if path_out is not None:
            comid_network = self.reach_generator.comid_network
            FM.save_data(
                comid_network,
                path_output=path_out,
                file_name="comid_network.hdf5",
            )
            linking_generator = copy.deepcopy(
                self.reach_generator.linking_network
            )
            linking_generator = linking_generator.reset_index()
            FM.save_data(
                linking_generator,
                path_output=path_out,
                file_name=f"linking_network.{extension}",
            )
        return

    def load_linking_network(self, path_file: str) -> None:
        """
        Description:
        ------------
            Load the linking network from a file.
        ________________________________________________________________________

        Args:
        -----
        :param path_file: str,
            Path to the file with the linking network
        :type path_file: str
        :return: None
        """

        if self.reach_generator is None:
            raise ValueError(
                "The reach generator has not been created yet"
                "please run the method `add_files` first"
            )
        self.reach_generator.linking_network = FM.load_data(
            path_file, pandas_dataframe=True
        )

        # Set index to comid id
        comid_id = self.reach_generator.comid_id
        self.reach_generator.linking_network.set_index(comid_id, inplace=True)
        linking_network = self.reach_generator.linking_network
        headwaters_comid = linking_network[linking_network["startflag"] == 1]
        headwaters_comid = headwaters_comid[
            headwaters_comid["extracted_comid"] == 1
        ]
        headwaters_comid = headwaters_comid.index.values
        self.reach_generator.exteracted_comids = headwaters_comid
        return

    def load_huc_list_in_comid_network(self, path_file: str) -> list:
        """
        Description:
        -------------
            Load the comid extracted in order to map the complete network.

        ________________________________________________________________________

        Args:
        -----
        :param path_file: str,
            Path to the comid_network file.
        :type path_file: str
        :param extract_all: bool, default=True,
            If True, the comid_network will be loaded and the complete
            comid lists will be extracted. If False, only the headwaters comid
            will be loaded.
        :return extracted_comids: list,
            List of extracted comids
        :rtype: list
        """
        comid_network = FM.load_data(path_file, keys=["huc_list"])
        return comid_network["huc_list"]

    def load_extracted_in_comid_network(self, path_file: str, huc=None) -> list:
        """
        Description:
        -------------
            Load the comid extracted in order to map the complete network.

        ________________________________________________________________________

        Args:
        -----
        :param path_file: str,
            Path to the comid_network file.
        :type path_file: str
        :param extract_all: bool, default=True,
            If True, the comid_network will be loaded and the complete
            comid lists will be extracted. If False, only the headwaters comid
            will be loaded.
        :return extracted_comids: list,
            List of extracted comids
        :rtype: list
        """
        if huc is None:
            comid_network = FM.load_data(path_file)
            huc = comid_network["huc_list"]
        elif isinstance(huc, str):
            huc = [huc]
            # Only extract a portion of the file
            comid_network = FM.load_data(path_file, keys=huc)
            comid_network["huc_list"] = huc
        headwaters_comid = {}
        for hu in huc:
            headwaters_comid[hu] = comid_network[hu]["comid_start"]
        self.reach_generator.comid_network = comid_network
        # if extract_all:
        #     comid_network = FM.load_data(path_file)
        #     self.reach_generator.comid_network = comid_network
        #     if huc is None:
        #         huc = comid_network['huc_list']
        #     elif isinstance(huc, str):
        #         huc = [huc]
        #     headwaters_comid = {}
        #     for hu in huc:
        #         headwaters_comid[hu] = comid_network[hu]['comid_start']
        # else:
        #     headwaters_comid = FM.load_data(path_file, keys=['comid_start'])
        #     headwaters_comid = headwaters_comid['comid_start']
        # utl.toc(time1)
        return headwaters_comid

    def get_reaches_from_network(
        self,
        huc: str,
        headwaters_comid: Union[list, np.ndarray] = None,
        linking_network_file: str = None,
        comid_network_file: str = None,
        min_distance: float = 1000.0,
        path_out: str = None,
        joblib_n_jobs: int = 1,
        file_name: Union[None, str] = None,
    ) -> None:
        """
        Description:
        -------------
            Get reach coordinates from the comid network dictionary.
        ________________________________________________________________________

        Args:
        -----
        :param headwaters_comid: list, np.ndarray,
            List of headwaters comids
        :type headwaters_comid: list, np.ndarray
        :param linking_network_file: str, None,
            Path to the linking network file
        :type linking_network_file: str, None
        :param comid_network_file: str, None,
            Path to the comid network file
        :type comid_network_file: str, None
        :param min_distance: float,
            Minimum distance of the reach from the headwaters in meters
        :type min_distance: float
        :param path_out: str, None,
            Path to save the reach coordinates
        :type path_out: str, None
        :param joblib_n_jobs: int, default=1,
            Number of jobs to run in parallel
        :type joblib_n_jobs: int
        :param extract_all: bool, default=True,
            If True, the comid_network will be loaded and the complete
            comid lists will be extracted. If False, only the headwaters comid
            will be loaded.
        :return: None
        """
        # --------------------------
        # keys
        # --------------------------
        keys = ["s", "x", "y", "z", "comid", "so", "within_waterbody"]
        keys_lab = {i: i for i in keys}
        # --------------------------
        # Check information
        # --------------------------
        # Check reach generator
        if self.reach_generator is None:
            raise ValueError(
                "The reach generator has not been created yet"
                "please run the method `add_files` first"
            )
        # Check linking network
        if linking_network_file is not None:
            self.load_linking_network(linking_network_file)
        linking_network = self.reach_generator.linking_network
        if linking_network is None:
            raise ValueError(
                "The linking network has not been created yet"
                "please run the method `map_network` first"
            )
        loading_from_file = False
        # Check comid network
        comid_network = self.reach_generator.comid_network
        try:
            huc_list = comid_network["huc_list"]
        except KeyError:
            raise ValueError(
                "comid_network not loaded. Please"
                " run load_extracted_in_comid_network first"
            )

        try:
            comid_network = copy.deepcopy(
                self.reach_generator.comid_network[huc]
            )
        except KeyError:
            raise KeyError(f"HUC {huc} not found in the comid network.")
        # Check for headwaters comids
        if headwaters_comid is None:
            headwaters_comid = comid_network["start_comid"]
        # --------------------------
        # get reach coordinates
        # --------------------------
        time1 = time.time()
        data_to_save = {
            str(hw): {
                key: []
                for key in keys + ["huc04", "huc_n", "start_comid", "uid"]
            }
            for hw in headwaters_comid
        }

        if joblib_n_jobs != 1:
            results_all = Parallel(n_jobs=joblib_n_jobs)(
                delayed(self._get_reach_from_network)(
                    hw,
                    min_distance,
                    linking_network,
                    comid_network,
                    loading_from_file,
                    comid_network_file,
                )
                for hw in headwaters_comid
            )
        else:
            results_all = [0] * len(headwaters_comid)
            for i_hw, hw in enumerate(headwaters_comid):
                time1 = time.time()
                results = self._get_reach_from_network(
                    hw,
                    min_distance,
                    linking_network,
                    comid_network,
                    loading_from_file,
                    comid_network_file,
                )
                results_all[i_hw] = results
                # print('Total time of method')
                # utl.toc(time1)
                # print('here')

        data_to_save = {}
        for i_hw, hw in enumerate(headwaters_comid):
            if results_all[i_hw][str(hw)]["start_comid"] == -1:
                continue
            data_to_save.update(results_all[i_hw])
        # Save Informaton
        if path_out is not None:
            if file_name is not None:
                FM.save_data(data_to_save, path_out, file_name=file_name)
            else:
                FM.save_data(
                    data_to_save,
                    path_out,
                    file_name=f"river_network_huc_{huc}.hdf5",
                )
        else:
            return data_to_save
        return

    def _get_reach_from_network(
        self,
        hw,
        min_distance,
        linking_network,
        comid_network,
        loading_from_file,
        comid_network_file,
    ):
        """ """
        # --------------------------
        # keys
        # --------------------------
        keys = [
            "s",
            "x",
            "y",
            "z",
            "comid",
            "so",
            "da_sqkm",
            "w_m",
            "within_waterbody",
        ]
        keys_lab = {i: f"{i}_o" for i in keys}
        # Loading from file
        time1 = time.time()
        if loading_from_file:
            comid_network = FM.load_data(comid_network_file, keys=[str(hw)])
        comid_list = list(comid_network[str(hw)])
        # print('Loading Network')
        # utl.toc(time1)

        # --------------------------
        # Verify reach length
        # --------------------------
        lengthkm = self.reach_generator.data_info.loc[
            comid_list, "lengthkm"
        ].values
        total_length = np.sum(lengthkm)
        remove = 0
        i_rep = 0
        cut = 10
        while total_length * 1000 < min_distance:
            additional_comid = linking_network.loc[
                comid_list[-1], "linking_comid"
            ]
            # print(i_rep, additional_comid)
            if (
                additional_comid == 0
                or additional_comid == "0"
                or i_rep >= cut
                or (additional_comid in comid_list)
            ):
                remove = 1
                break
            comid_list.append(additional_comid)
            lengthkm = self.reach_generator.data_info.loc[
                comid_list, "lengthkm"
            ].values
            total_length = np.sum(lengthkm)
            i_rep += 1
        # print('Extending reach')
        # utl.toc(time1)

        # --------------------------
        # Extract original coordinates
        # --------------------------
        time1 = time.time()
        data = self.reach_generator.map_coordinates(
            comid_list, file_coords=self.coords_file
        )
        # print('Extracting original coordinates')
        # utl.toc(time1)

        # --------------------------
        # Remove Reaches
        # --------------------------
        delta_time_extract = time.time() - time1
        if remove == 1 or len(data["x"].values) <= 3:
            remove = 0
            return {str(hw): {"start_comid": -1}}

        huc_n = linking_network.loc[hw, "huc_n"]
        data["comid"] = data.index.values
        comid_val = data["comid"].values
        within_waterbody = linking_network.loc[:, "within_waterbody"]
        data["within_waterbody"] = within_waterbody.loc[comid_val].values

        # Save data into dict
        time1 = time.time()
        # data = {str(hw): {
        #     key: data[keys_lab[key]].values for key in keys}}
        data = {str(hw): {keys_lab[key]: data[key].values for key in keys}}
        data[str(hw)]["huc04"] = self.huc04
        data[str(hw)]["huc_n"] = huc_n
        data[str(hw)]["start_comid"] = hw
        data[str(hw)]["time_extract"] = delta_time_extract
        # Add to data
        # print('Storing Information')
        # utl.toc(time1)
        return data

    def get_reach(
        self,
        id_value,
        start_comid=None,
        comid_list=None,
        huc=8,
        kwargs_resample={"method": "geometric_mean"},
        scale_by_width=False,
        uid=None,
    ):
        """
        Description:
        -------------
            Get complete reach.
        ________________________________________________________________________

        Args:
        ------------
        :param id_value: str, float, int
            id of the river.
        :type id_value: int
        :param start_comid: float,
            comid of the start of the reach.
        :type start_comid: float
        :param comid_list: list
            list of comids to get the reach.
        :param huc: int
            huc number to get reach, if 4 it will get to the subregion level.
        :type huc: int
        :param kwargs_resample: dict
            kwargs to resample the reach. by default the method is
            'geometric_mean'. For more information look at the River class.
        :type kwargs_resample: dict
        :param scale_by_width: bool, default=False
            If True, the reach will be scaled by the width of the river.
        :type scale_by_width: bool
        :param uid: str, optional,
            Unique ID of the river
        :type uid: str
        """
        # --------------------------
        # Check if ID exists
        # --------------------------
        try:
            self.id_values.index(id_value)
            self.logger.warning(f"ID {id_value} already exists. Ovewriting")
        except ValueError:
            pass
        # --------------------------
        # keys
        # --------------------------
        keys = ["s", "x", "y", "z", "comid", "so", "da_sqkm", "w_m"]
        keys_lab = {i: i for i in keys}
        # --------------------------
        # Generate Reach
        # --------------------------
        if start_comid is None and comid_list is None:
            raise ValueError(
                "Either start_comid or comid_list must be" " provided"
            )
        reach_generator = self.reach_generator
        if reach_generator is None:
            raise ValueError(
                "No reach generator defined."
                " Please run self.add_files() first"
            )
        # Get COMID network
        self.logger.info("Getting COMID network")
        if start_comid is not None:
            comid_network, _ = reach_generator.map_complete_reach(
                start_comid, huc, do_not_overlap=False
            )
        elif comid_list is None:
            comid_network = comid_list
        # Get data
        self.logger.info("Mapping coordinates")
        data_pd = reach_generator.map_coordinates(
            comid_network, file_coords=self.coords_file
        )
        comid = np.array(data_pd.index)
        data = {}
        for key in list(data_pd):
            data[key] = data_pd[key].values

        within_waterbody = reach_generator.data_info.loc[
            comid, "within_waterbody"
        ].values

        self.logger.info("Adding River")
        self.add_river(
            id_value,
            data[keys_lab["x"]],
            data[keys_lab["y"]],
            s=data[keys_lab["s"]],
            z=data[keys_lab["z"]],
            so=data[keys_lab["so"]],
            da_sqkm=data[keys_lab["da_sqkm"]],
            w_m=data[keys_lab["w_m"]],
            comid=comid,
            huc04=self.huc04,
            huc_n=huc,
            start_comid=start_comid,
            uid=uid,
            kwargs_resample=kwargs_resample,
            within_waterbody=within_waterbody,
            scale_by_width=scale_by_width,
        )
        return

    @staticmethod
    def _compile_database(database_dict):
        for i, id_value in enumerate(list(database_dict)):
            if i == 0:
                database = database_dict[id_value]
            else:
                database = pd.concat([database, database_dict[id_value]])
        # reset index
        database = database.reset_index(drop=True)
        return database

    def get_metric_databases(self):
        """
        Description:
        ------------
            Get metrics from the meander databases.
        ________________________________________________________________________
        """
        database = pd.DataFrame
        for i, id_value in enumerate(self.id_values):
            if i == 0:
                database = copy.deepcopy(self.rivers[id_value].database)
            else:
                database = pd.concat([database, self.rivers[id_value].database])
        database.reset_index(drop=False, inplace=True)
        return database

    def save_databases_meanders(self, path_output, file_name):
        """
        Description:
        ------------
            Save meander database
        ________________________________________________________________________

        Args:
        ------------
        :param path_output: str,
            Path to save the database
        :type path_output: str
        :param file_name: str,
            Name of the file to save the database
        :type file_name: str
        """
        database = self.get_metric_databases()
        FM.save_data(database, path_output=path_output, file_name=file_name)
        return

    def save_tree_scales(
        self,
        path_output,
        fn_tree_scales="tree_scales.p",
        fn_tree_scales_database="tree_scales_database.feather",
        rivers_ids=[],
    ):
        """
        Description:
        ------------
            Save meander database
        ________________________________________________________________________

        Args:
        ------------
        :param path_output: str,
            Path to save the database
        :type path_output: str
        :param file_name: str,
            Name of the file to save the Tree scales object
        :type file_name: str
        """
        tree_scales = {}
        tree_scales_database = {}

        for i_key, key in enumerate(self.rivers["id_values"]):
            if len(rivers_ids) > 0 and key not in rivers_ids:
                continue

            tree_scale = None
            tree_scale = self.rivers[key].tree_scales

            # Only save trees with information
            if tree_scale is not None:
                tree_scales[str(key)] = self.rivers[key].tree_scales.trees
                tree_scales_database[key] = self.rivers[
                    key
                ].tree_scales_database
            else:
                self.logger.info(f"{key} No tree scales extracted")
                # print('No tree scales extracted')
                continue

        # Save tree scales
        if len(tree_scales) > 0:
            FM.save_data(
                tree_scales, path_output=path_output, file_name=fn_tree_scales
            )
            database = self._compile_database(tree_scales_database)
            FM.save_data(
                database,
                path_output=path_output,
                file_name=fn_tree_scales_database,
            )
        return

    def extract_cwt_data(self, rivers_ids=[]):
        """
        Description:
        ------------
            Extract continuous wavelet transform information.
        """
        cwt_info = {}

        for i_key, key in enumerate(self.rivers["id_values"]):
            if len(rivers_ids) > 0 and key not in rivers_ids:
                continue

            w_m = self.rivers[key].w_m
            w_m_gm = self.rivers[key].w_m_gm
            x = self.rivers[key].x
            y = self.rivers[key].y
            c = self.rivers[key].c
            s = self.rivers[key].s
            angle = self.rivers[key].angle
            comid = self.rivers[key].comid
            if comid is None:
                comid = [""] * len(x)
            wave_c = self.rivers[key].cwt_wave_c
            wavelength_c = self.rivers[key].cwt_wavelength_c
            scales_c = self.rivers[key].cwt_scales_c
            power_c = self.rivers[key].cwt_power_c
            coi_c = self.rivers[key].cwt_coi_c
            gws_c = self.rivers[key].cwt_gws_c
            sawp_c = self.rivers[key].cwt_sawp_c
            power_sig = self.rivers[key].cwt_power_c_sig
            gws_c_sig = self.rivers[key].cwt_gws_c_sig
            sawp_c_sig = self.rivers[key].cwt_sawp_c_sig

            wave_angle = self.rivers[key].cwt_wave_angle
            wavelength_angle = self.rivers[key].cwt_wavelength_angle
            scales_angle = self.rivers[key].cwt_scales_angle
            power_angle = self.rivers[key].cwt_power_angle
            coi_angle = self.rivers[key].cwt_coi_angle
            gws_angle = self.rivers[key].cwt_gws_angle
            sawp_angle = self.rivers[key].cwt_sawp_angle
            power_sig_angle = self.rivers[key].cwt_power_angle_sig
            gws_angle_sig = self.rivers[key].cwt_gws_angle_sig
            sawp_angle_sig = self.rivers[key].cwt_sawp_angle_sig

            # Save Information
            cwt_info.update(
                {
                    key: {
                        "x": x,
                        "y": y,
                        "c": c,
                        "s": s,
                        "angle": angle,
                        "w_m": w_m,
                        "comid": comid,
                        "w_m_gm": w_m_gm,
                        "wave_c": wave_c,
                        "wavelength_c": wavelength_c,
                        "scales_c": scales_c,
                        "power_c": power_c,
                        "coi_c": coi_c,
                        "gws_c": gws_c,
                        "sawp_c": sawp_c,
                        "power_c_sig": power_sig,
                        "gws_c_sig": gws_c_sig,
                        "sawp_c_sig": sawp_c_sig,
                        "wave_angle": wave_angle,
                        "wavelength_angle": wavelength_angle,
                        "scales_angle": scales_angle,
                        "power_angle": power_angle,
                        "coi_angle": coi_angle,
                        "gws_angle": gws_angle,
                        "sawp_angle": sawp_angle,
                        "power_angle_sig": power_sig_angle,
                        "gws_angle_sig": gws_angle_sig,
                        "sawp_angle_sig": sawp_angle_sig,
                    }
                }
            )

        return cwt_info

    def save_cwt_data(self, path_output, file_name, rivers_ids=[]):
        """
        Description:
        ------------
            Save continuous wavelet transform information.
        """
        cwt_info = self.extract_cwt_data(rivers_ids=rivers_ids)
        if len(cwt_info) > 0:
            FM.save_data(cwt_info, path_output=path_output, file_name=file_name)
        return

    def save_rivers(
        self,
        path_output,
        file_name,
        save_cwt_info=False,
        rivers_ids=[],
        fn_tree_scales="tree_scales.p",
        fn_tree_scales_database="tree_scales_database.feather",
        fn_meander_database="meander_database.csv",
    ):
        """
        Description:
        ------------
            Save river files.
        ________________________________________________________________________

        Args:
        ------------
        :param path_output: str,
            Path to save the database
        :type path_output: str
        :param file_name: str,
            Name of the file to save the database
        :type file_name: str
        :param save_cwt_info: bool, default=False,
            If True, the CWT information will be saved as a pickle file.
        :type save_cwt_info: bool
        :param fn_tree_scales: str, default='tree_scales.p',
            Name of the file to save the tree scales, this file must be saved as
            a pickle file.
        :type fn_tree_scales: str
        :param fn_tree_scales_database: str,
            default='tree_scales_database.feather',
            Name of the file to save the tree scales database, this file can be
            saved as a feather or csv file.
        :type fn_tree_scales_database: str
        :param fn_meander_database: str, default='meander_database.csv',
            Name of the file to save the meander database, this file can be
            saved as a feather or csv file.
        :type fn_meander_database: str
        """
        extension = file_name.split(".")[-1]
        if extension in ("pickle", "p"):
            data_save = self.rivers
            FM.save_data(
                data_save, path_output=path_output, file_name=file_name
            )
        elif extension in ("hdf5", "h5"):
            data_save = {}
            cwt_poly = {}
            cwt_zc_lines = {}
            tree_scales = {}
            tree_scales_database = {}

            for i_key, key in enumerate(self.rivers["id_values"]):
                if len(rivers_ids) > 0 and key not in rivers_ids:
                    continue

                self.logger.info(f"Saving {key}")
                tree_scale = None
                data_save[str(key)] = self.rivers[key].extract_data_to_save()
                # Remove variables that cannot be saved in hdf5
                try:
                    data_save[str(key)].pop("cwt_poly")
                    data_save[str(key)].pop("cwt_zc_lines")
                    cwt_poly[str(key)] = self.rivers[key].cwt_poly
                    cwt_zc_lines[str(key)] = self.rivers[key].cwt_zc_lines
                except KeyError:
                    self.logger.info(f"{key} No CWT poly or ZC lines extracted")

                try:
                    data_save[str(key)].pop("tree_scales")
                    data_save[str(key)].pop("tree_scales_database")
                    data_save[str(key)].pop("tree_scales_database_meanders")
                    tree_scale = self.rivers[key].tree_scales
                except KeyError:
                    self.logger.info(f"{key} No tree scales extracted")

                try:
                    data_save[str(key)].pop("meanders")
                    data_save[str(key)].pop("database")
                except KeyError:
                    self.logger.info(f"{key} No meanders extracted to save")

                try:
                    data_save[str(key)].pop("splines")
                except KeyError:
                    self.logger.info(f"{key} No splines included")

                # Only save trees with information
                if tree_scale is not None:
                    tree_scales[str(key)] = self.rivers[key].tree_scales.trees
                    tree_scales_database[key] = self.rivers[
                        key
                    ].tree_scales_database

            # Save Information
            FM.save_data(
                data_save, path_output=path_output, file_name=file_name
            )
            # Save tree scales
            if len(tree_scales) > 0:
                FM.save_data(
                    tree_scales,
                    path_output=path_output,
                    file_name=fn_tree_scales,
                )
                database = self._compile_database(tree_scales_database)
                FM.save_data(
                    database,
                    path_output=path_output,
                    file_name=fn_tree_scales_database,
                )

            # Save CWT information
            if save_cwt_info and len(cwt_poly) > 0:
                file_name = "cwt_poly.p"
                FM.save_data(
                    cwt_poly, path_output=path_output, file_name=file_name
                )
                file_name = "cwt_zc_lines.p"
                FM.save_data(
                    cwt_zc_lines, path_output=path_output, file_name=file_name
                )

            database = self.get_metric_databases()
            if len(database) > 0:
                self.save_databases_meanders(path_output, fn_meander_database)

        return

    def load_river_network_ids(self, path_file: str):
        river_file = FM.load_data(path_file)
        id_values = list(river_file)
        return id_values

    def load_river_network(
        self,
        path_file: str,
        headwaters_comid: Union[str, None] = None,
        fn_tree_scales: Union[str, None] = None,
        fn_tree_scales_database: Union[str, None] = None,
        fn_meanders_database: Union[str, None] = None,
        kwargs_resample=None,
        scale_by_width=None,
        recalculate_inflection_points=False,
        **kwargs,
    ) -> None:
        """
        Description:
        -------------
            Load the river network from an hdf5 file.
        ________________________________________________________________________

        Args:
        -----
        :param path_file: str,
            Path to the hdf5 file
        :type path_file: str
        :param headwaters_comid: str, None, Default is None
            Name of the headwaters to extract. If None all the headwaters
            will be extracted.
        :type headwaters_comid: str, Defualt is None
        :param fn_tree_scales: str, Default is None
            Name of the file with the tree scales.
        :type fn_tree_scales: str
        :param fn_tree_scales_database: str, Default is None
            Name of the file with the tree scales database.
        :type fn_tree_scales_database: str
        :param fn_meanders_database: str, Default is None
            Name of the file with the meanders database.
        :type fn_meanders_database: str
        :param kwargs_resample: dict, Default is None
            Dictionary with the kwargs to resample the reach. If None, the
            kwargs will be extracted from the file.
        :type kwargs_resample: dict
        :param scale_by_width: bool,
            Have the rivers to be scaled by the minimum width.
        :type scale_by_width: bool
        :param kwargs: dict,
            Additional kwargs to add to the river.
        :type kwargs: dict
        """
        # Check headwaters
        if headwaters_comid is not None:
            if (
                isinstance(headwaters_comid, str)
                or isinstance(headwaters_comid, int)
                or isinstance(headwaters_comid, float)
            ):
                headwaters_comid = [headwaters_comid]
            headwaters_comid_str = [str(i) for i in headwaters_comid]
            data_rivers = FM.load_data(path_file, keys=headwaters_comid_str)
        else:
            data_rivers = FM.load_data(path_file)

        # Load tree_scales if available
        if fn_tree_scales is not None:
            tree_scales_data = FM.load_data(fn_tree_scales)

        if fn_tree_scales_database is not None:
            tree_scales_database = FM.load_data(
                fn_tree_scales_database, pandas_dataframe=True
            )

        if fn_meanders_database is not None:
            meanders_database = FM.load_data(
                fn_meanders_database, pandas_dataframe=True
            )

        # Load cwt_trees_file if available
        # if cwt_trees_file is not None:
        #     cwt_trees = FM.load_data(cwt_trees_file)

        # Load data
        i = 0
        for hw in list(data_rivers):
            try:
                hw_r = float(hw)
            except ValueError:
                hw_r = hw
            if str(hw_r) != hw:
                hw_r = hw

            data_river = data_rivers[hw]
            if fn_tree_scales is not None:
                # Check if tree_scales exists if not continue to next river
                try:
                    tree_scales = RiverTreeScales(tree_scales_data[hw])
                    add_tree_scales = True
                except KeyError:
                    add_tree_scales = False
            # See if data has within_waterbody information
            try:
                within_waterbody = data_river["within_waterbody_o"]
            except KeyError:
                within_waterbody = None
            huc04 = data_river["huc04"]

            try:
                start_comid = float(hw)
            except:
                start_comid = None

            if scale_by_width is None:
                try:
                    scale_by_width = data_river["scale_by_width"]
                except KeyError:
                    scale_by_width = False

            if kwargs_resample is None:
                try:
                    kw_resample = data_river["kwargs_resample"]
                except KeyError:
                    kw_resample = {}
            else:
                try:
                    kw_resample = kwargs_resample[hw]
                except KeyError:
                    kw_resample = {}

            # print(kw_resample)
            # Add River Information
            self.add_river(
                hw_r,
                data_river["x_o"],
                data_river["y_o"],
                s=data_river["s_o"],
                z=data_river["z_o"],
                so=data_river["so_o"],
                comid=data_river["comid_o"],
                da_sqkm=data_river["da_sqkm_o"],
                w_m=data_river["w_m_o"],
                huc04=huc04,
                huc_n=data_river["huc_n"],
                start_comid=start_comid,
                within_waterbody=within_waterbody,
                scale_by_width=scale_by_width,
                kwargs_resample=kw_resample,
                **kwargs,
            )
            # Include additional data
            for key in list(data_river):
                if key not in [
                    "s",
                    "x",
                    "y",
                    "z",
                    "so",
                    "comid",
                    "da_sqkm",
                    "w_m",
                    "huc04",
                    "huc_n",
                ]:
                    try:
                        self.rivers[hw_r].__dict__[key] = data_river[key]["0"]
                    except:
                        self.rivers[hw_r].__dict__[key] = data_river[key]
                    if isinstance(self.rivers[hw_r].__dict__[key], bytes):
                        self.rivers[hw_r].__dict__[key] = (
                            self.rivers[hw_r].__dict__[key].decode("utf-8")
                        )
            # Add Tree scales
            if fn_tree_scales is not None and add_tree_scales:
                tree_scales = RiverTreeScales(tree_scales_data[hw])
                self.rivers[hw_r].tree_scales = copy.deepcopy(tree_scales)
                self.rivers[hw_r].update_tree_scales_meander_database()
                # self.rivers[hw_r].tree_scales_database = \
                #     self.rivers[hw_r].tree_scales.compile_database()
            # Add tree scales database
            if fn_tree_scales_database is not None and add_tree_scales:
                start_comid = self.rivers[hw_r].start_comid
                # Clip database
                tree_scales_d_river = tree_scales_database[
                    tree_scales_database["start_comid"] == start_comid
                ]
                # Add database
                self.rivers[hw_r].tree_scales_database = copy.deepcopy(
                    tree_scales_d_river
                )

            self.rivers[hw_r].id_meanders = []
            if fn_meanders_database is not None:
                database = meanders_database[
                    meanders_database["start_comid"] == hw_r
                ]

                if len(database) == 0:
                    database = meanders_database[
                        meanders_database["id_reach"] == self.rivers[hw_r].uid
                    ]

                # Add meanders
                self.rivers[hw_r].id_meanders = []
                if len(database) > 0:
                    database.reset_index(drop=True, inplace=True)
                    for i in range(len(database)):
                        idx_start = int(database.loc[i, "idx_start"])
                        idx_end = int(database.loc[i, "idx_end"])
                        id_meander = int(database.loc[i, "id_meander"])
                        sk = database.loc[i, "skewness"]
                        fl = database.loc[i, "flatness"]
                        automatic_flag = database.loc[i, "automatic_flag"]
                        try:
                            inflection_flag = database.loc[i, "inflection_flag"]
                        except KeyError:
                            inflection_flag = False

                        if recalculate_inflection_points:
                            x_inf = None
                            y_inf = None
                            s_inf = None
                        else:
                            try:
                                x_inf = database.loc[i, "x_inf"]
                                y_inf = database.loc[i, "y_inf"]
                                s_inf = database.loc[i, "s_inf"]
                            except KeyError:
                                x_inf = None
                                y_inf = None
                                s_inf = None

                            # Change values from string to lists
                            if isinstance(x_inf, str):
                                if (
                                    x_inf == ""
                                    or x_inf == "nan"
                                    or x_inf == "None"
                                ):
                                    x_inf = None
                                    y_inf = None
                                    s_inf = None
                                else:
                                    x_inf = x_inf.replace("nan", "np.nan")
                                    y_inf = y_inf.replace("nan", "np.nan")
                                    s_inf = s_inf.replace("nan", "np.nan")
                                    # find is separation is by comma
                                    if "," not in x_inf:
                                        x_inf = x_inf.replace(" ", ",")
                                    if "," not in y_inf:
                                        y_inf = y_inf.replace(" ", ",")
                                    if "," not in s_inf:
                                        s_inf = s_inf.replace(" ", ",")
                                    x_inf = np.array(eval(x_inf))
                                    y_inf = np.array(eval(y_inf))
                                    s_inf = np.array(eval(s_inf))

                            if x_inf is not None:
                                if np.sum(np.isnan(x_inf)) >= 1:
                                    x_inf = None
                                    y_inf = None
                                    s_inf = None

                        try:
                            tree_id = database.loc[i, "tree_id"]
                        except KeyError:
                            tree_id = -1
                        self.rivers[hw_r].add_meander(
                            id_meander,
                            idx_start,
                            idx_end,
                            sk=sk,
                            fl=fl,
                            automatic_flag=automatic_flag,
                            inflection_flag=inflection_flag,
                            tree_id=tree_id,
                            x_inf=x_inf,
                            y_inf=y_inf,
                            s_inf=s_inf,
                        )

            # TODO: Correct the loading of poly and zc_lines
            # Add Tree Scales database
            # if cwt_trees_file is not None:
            #     self.rivers[float(hw)].cwt_poly[0] = \
            #         cwt_trees[hw]['cwt_poly'][0]
            #     self.rivers[float(hw)].cwt_zc_lines[0] = \
            #         cwt_trees[hw]['cwt_zc_lines'][0]

        # Sort Rivers from longest to shortest
        id_rivers = self.rivers["id_values"]
        s_final = [self.rivers[i].s[-1] for i in id_rivers]
        argsort = np.argsort(s_final)[::-1]
        self.rivers["id_values"] = [id_rivers[i] for i in argsort]
        self.id_values = self.rivers["id_values"]

        return

    def plot_rivers(
        self,
        comids: Union[float, int, str, list, None] = None,
        engine: str = "matplotlib",
        data_source="resample",
        **kwargs,
    ):
        """
        Description:
        -------------
            Plot rivers from the river network.
        ________________________________________________________________________

        Args:
        -----
        :param comid: float, int, str, list
            Comid of the river to plot
        :type comid: float, int, str
        """
        if comids is not None:
            if (
                isinstance(comids, float)
                or isinstance(comids, int)
                or isinstance(comids, str)
            ):
                comids = [comids]
        else:
            comids = self.rivers["id_values"]

        if engine == "matplotlib":
            fig, ax = graphs.plot_rivers_matplotlib(
                self, comids, data_source, **kwargs
            )
        elif engine == "plotly":
            fig = graphs.plot_rivers_plotly(self, comids, data_source, **kwargs)
        return fig


class RiverTransect:
    """
    This class is the basic form of rivers. Each river is contains the following
    base attributes.

    ===================== =====================================================
    Attribute             Description
    ===================== =====================================================
    uid                   Unique Identifier for the River.
    id_value              id for river.
    s_o                   Original vector of distances of each point.
    x_o                   Original x-dir coordiantes.
    y_o                   Original y-dir coordiantes.
    z_o                   Original height of each point.
    so_o                  Original Stream order of each point.
    comid_so              Original comid for rivers
    s                     Spline fitted vector of distances of each point.
    x                     Spline fitted x-dir coordiantes.
    y                     Spline fitted y-dir coordiantes.
    z                     Spline fitted Height of each point.
    so                    Spline fitted stream order of each point.
    comid                 comid for rivers
    da_sqkm               Drainage area in square kilometers. Optional
    w_m                   Width of the river in meters. Optional
    huc04                 HUC 04 for the reach. Optional
                          This value is used for future reconstruction of the
                          river using the NHDPlus dataset.
    huc_n                 number of HUC where data was clipped. Optional
                          This value is used for future reconstruction of the
                          river using the NHDPlus dataset.
    start_comid           comid of the start of the reach. Optional
                          This value is used for future reconstruction of the
                          river using the NHDPlus dataset.
    within_waterbody      vector that provides the section where the river is
                          within a water body. Optional
    resample_flag         Flag to resample the river. Default is True.
    kwargs_resample       Resample kwargs. Default is None.
                          The values accepted are:
                            'method': str, method to calculate the distance
                                between points. Options include:
                                'min', 'mean', 'geometric_mean',
                                'geometric_mean_width', 'min_width'.
                                The shorter the distance the heavier the file
                                will be.
                                Default is 'min_width'.
                            'ds': Set distance to perorm the resampling.
                                If this value is None the distance will be
                                calculated using the method provided. This
                                distance will overide the method provided.
                                The shorter the distance the heavier the file
                                will be.
                                Default is None.
                            'k': int, polynomial order of the spline.
                                Default is 3.
                            'smooth': float, smoothing factor for the spline.
                                The smoothing factor depends on the amount of
                                points in the data, then the given number will
                                be multiplied by the length of the data. Please
                                look at `scipy.interpolate.UnivariateSpline` for
                                more information. We recommend testing values
                                that vary over orders of magnitude with respect
                                to the length of the data. Default is 0.0.
                            'ext': int, controls the extrapolation mode for
                                elements not in the interval defined by the
                                knot sequence. Default is 0.
                                0: return extrapolated values.
                                1: return zero
                                2: return 0 raise ValueError
                                3: return the boundary value.
                                Please look at
                                `scipy.interpolate.UnivariateSpline` for more
                                information.
    logger                Logger object.
    ===================== =====================================================

    The following are the methods of the class.

    =================================== =====================================================
    Methods                             Description
    =================================== =====================================================
    set_gamma_width                     Set value of gamma threshold for width for the CWT.
    set_data_source                     Set data source to extract data from
    extract_data_to_save                Extract data to save in a file.
    set_planimetry_derivatives          Set planimetry derivatives.
    set_splines                         Set splines of the x and y coordinates of the river.
    eval_splines                        Evaluate splines of the x and y coordinates of the river.
    scale_coordinates                   Scale coordinates by a given value
    translate_xy_start                  Translate coordinates to xy start
    calculate_spline                    Calculate spline of the river
    calculate_smooth                    Smooth the planimetry of the river.
    calculate_curvature                 Calculate curvature of the river.
    extract_cwt_tree                    Extract CWT and tree of the river
    extract_meanders                    Extract meanders from the river with CWT process.
    get_cwt_curvature                   Get CWT of the curvature of the river.
    get_cwt_angle                       Get CWT of the angle of the river.
    extract_tree                        Extract tree from the cwt of the river.
    find_peaks_in_poly                  Find peaks in the polygons of the cwt.
    detect_meander_from_cwt             Detect meander from the cwt.
    get_tree_center_points_in_planimtry Get center points of the tree.
    update_tree_scales                  Update tree scales object
    get_meander_bounds_from_cwt         Get meander bounds from the cwt.
    add_meanders_from_bounds            Add meanders from captured bounds comming from cwt.
    plot_cwt                            Plot cwt tree of the river.
    add_meander                         Add meander to the river.
    report_meander_metrics              Report meander Metrics.
    remove_meander                      Remove meander from list.
    =================================== =====================================================
    """

    def __init__(
        self,
        uid: Union[str, int, float],
        id_value: Union[str, int, float],
        x: Union[list, np.ndarray],
        y: Union[list, np.ndarray],
        s: Union[list, np.ndarray],
        z: Union[List, np.ndarray, None] = None,
        so: Union[None, list, np.ndarray] = None,
        comid: Union[None, list, np.ndarray] = None,
        da_sqkm: Union[None, list, np.ndarray] = None,
        w_m: Union[None, list, np.ndarray] = None,
        huc04: Union[None, list, np.ndarray] = None,
        huc_n: Union[None, list, np.ndarray] = None,
        start_comid: Union[None, str, float, int] = None,
        within_waterbody: Union[None, list, np.ndarray] = None,
        resample_flag: Union[None, bool] = True,
        kwargs_resample: Union[None, dict] = None,
        scale_by_width: Union[None, bool] = False,
        logger: Union[None, Logger] = None,
    ):
        # -------------------
        # Attributes
        # -------------------
        assert len(x) == len(y), "x and y must have the same length"
        if logger is None:
            self._logging = logging.getLogger(self.__class__.__name__)
        else:
            self._logging = logger
        self.uid = uid
        self.id_value = id_value
        coords = np.array([x, y]).T
        if s is None:
            s = RF.get_reach_distances(coords)
        self.x_o = copy.deepcopy(x)
        self.y_o = copy.deepcopy(y)
        self.s_o = copy.deepcopy(s)
        self.scale = 1
        self.xy_start = [self.x_o[0], self.y_o[0]]
        self.x_start = self.x_o[0]
        self.y_start = self.y_o[0]
        self.translate = False
        self.planimetry_derivatives = None
        self.splines = None
        # --------------------------------
        # Set optional variables to NaN
        # --------------------------------
        if z is None:
            self.z_o = np.array([np.nan for _ in self.x_o])
        else:
            assert len(z) == len(x), "z must have the same length as x"
            self.z_o = z
        if so is None:
            self.so_o = np.array([np.nan for _ in self.x_o])
        else:
            assert len(so) == len(x), "so must have the same length as x"
            self.so_o = so
        if comid is None:
            self.comid_o = np.array(["0" for _ in self.x_o])
        else:
            assert len(comid) == len(x), "comid must have the same length as x"
            self.comid_o = comid
        if da_sqkm is None:
            self.da_sqkm_o = np.array([np.nan for _ in self.x_o])
        else:
            assert len(da_sqkm) == len(
                x
            ), "da_sqkm must have the same length as x"
            self.da_sqkm_o = da_sqkm
        if w_m is None:
            self.w_m_o = np.array([np.nan for _ in self.x_o])
        else:
            # assert len(w_m) == len(x), 'w_m must have the same length as x'
            self.w_m_o = w_m
        if within_waterbody is None:
            self.within_waterbody_o = np.zeros(len(self.x_o))
        else:
            assert len(within_waterbody) == len(
                x
            ), "within_waterbody must have the same length as x"
            self.within_waterbody_o = within_waterbody

        # Scale by width
        self.w_m_min = np.min(self.w_m_o)
        self.w_m_gm = 10 ** np.mean(np.log10(self.w_m_o))
        self.scale_by_width = scale_by_width
        self.scaled_data = False

        self.kwargs_resample_default = {
            "method": "min_width",
            "ds": 0,
            "k": 3,
            "smooth": 0.0,
            "ext": 0,
        }
        if np.isnan(self.w_m_gm):
            self.kwargs_resample_default["method"] = "geometric_mean"
        self.kwargs_resample = copy.deepcopy(self.kwargs_resample_default)
        if kwargs_resample is not None:
            self.kwargs_resample.update(kwargs_resample)
        # self.kwargs_resample['smooth'] *= len(self.x_o)
        # -------------------
        # Resample
        # -------------------
        if not (resample_flag):
            self.s = copy.deepcopy(self.s_o)
            self.x = copy.deepcopy(self.x_o)
            self.y = copy.deepcopy(self.y_o)
            self.z = copy.deepcopy(self.z_o)
            self.so = copy.deepcopy(self.so_o)
            self.comid = copy.deepcopy(self.comid_o)
            self.da_sqkm = copy.deepcopy(self.da_sqkm_o)
            self.w_m = copy.deepcopy(self.w_m_o)
            self.within_waterbody = copy.deepcopy(self.within_waterbody_o)
        else:
            # Calculate the spline
            self.calculate_spline(
                function=RF.fit_splines_complete, **self.kwargs_resample
            )
            # self.logger.info('Resample calculated with '
            #                  '`River.calculate_spline()`')

        if scale_by_width and not (np.isnan(self.w_m_gm)):
            # TODO: Include flag for exporting values to shapefiles, remember to unscale the data
            self.logger.info(" Scaling curvature by width")
            self.x /= self.w_m_gm
            self.y /= self.w_m_gm
            self.z /= self.w_m_gm
            self.s /= self.w_m_gm
            self.scaled_data = True

        # Calculate resample distance
        self.ds = np.mean(np.diff(self.s))
        # Check equal distance between s points
        dif = np.diff(self.s)
        if not (np.all(np.isclose(dif, self.ds, rtol=1e-2))):
            self.logger.warning(
                "Distance between points is not constant,"
                " please run `River.calculate_spline()`"
            )

        # -------------------
        # Other attributes
        # -------------------
        self.huc04 = huc04
        self.huc_n = huc_n
        self.start_comid = start_comid

        # -----------------------
        # Additional River Data
        # -----------------------
        self.gamma = 10
        self.gamma_w_m = self.gamma * self.w_m
        self.data_source = "resample"
        self.s_smooth = None
        self.x_smooth = None
        self.y_smooth = None
        self.ds_smooth = None
        self.w_m_smooth = None
        self.da_sqkm_smooth = None
        self.r = None
        self.c = None
        self.angle = None
        # -------------------------
        # CWT tree construction
        # -------------------------
        # --------------
        # Curvature
        # --------------
        # Wave information
        self.cwt_parameters_c = None
        self.cwt_wave_c = None
        self.cwt_wavelength_c = None
        self.cwt_scales_c = None
        self.cwt_coi_c = None
        self.cwt_power_c = None
        self.cwt_gws_c = None
        self.cwt_gws_peak_wavelength_c = None
        self.cwt_sawp_c = None
        self.cwt_signif_c = None
        self.cwt_sawp_c = None
        self.cwt_sig95_c = None
        self.cwt_power_c_sig = None
        self.cwt_gws_c_sig = None
        self.cwt_sawp_c_sig = None
        # Tree information
        self.cwt_conn = None
        self.cwt_regions = None
        self.cwt_poly = None
        self.cwt_zc_lines = None
        self.cwt_zc_sign = None
        self.cwt_peak_pwr = None
        self.cwt_peak_row = None
        self.cwt_peak_col = None
        self.cwt_meander_id = None
        self.cwt_ml_tree = None
        self.cwt_gws = None
        self.bound_to_poly = False
        self.tree_scales = None
        self.tree_scales_database = None
        self.cwt_planimetry_coords = None
        # -----------------
        # Direction Angle
        # -----------------
        self.cwt_parameters_angle = None
        self.cwt_wave_angle = None
        self.cwt_wavelength_angle = None
        self.cwt_scales_angle = None
        self.cwt_power_angle = None
        self.cwt_coi_angle = None
        self.cwt_gws_angle = None
        self.cwt_gws_peak_wavelength_angle = None
        self.cwt_sawp_angle = None
        self.cwt_signif_angle = None
        self.cwt_conn_angle = None
        self.cwt_sig95_angle = None
        self.cwt_power_angle_sig = None
        self.cwt_gws_angle_sig = None
        self.cwt_sawp_angle_sig = None
        # -------------------------
        # Meanders
        # -------------------------
        self.metrics_reach = {}
        self._calc_vars = CALC_VARS
        self.meanders = {}
        self.id_meanders = []
        self.tree_scales_database_meanders = None
        self.total_sinuosity = None
        # -------------------------
        # Report variables
        # -------------------------
        self._database_vars = [
            "id_reach",
            "huc04_n",
            "huc_n",
            "start_comid",
            "id_meander",
            "x_start",
            "x_end",
            "y_start",
            "y_end",
            "scale",
            "x_start",
            "y_start",
            "translate",
        ] + self._calc_vars
        self.database = {i: [] for i in self._database_vars}
        self.database = pd.DataFrame.from_dict(self.database)
        # self.database.set_index('id')
        return

    # --------------------
    # Getters
    # --------------------
    @property
    def logger(self):
        return self._logging

    @logger.setter
    def logger(self, logger):
        assert isinstance(self._logging, Logger)
        self._logging = logger

    # --------------------
    # Setters
    # --------------------
    def set_gamma_width(self, gamma: Union[None, float, int] = None):
        """
        Description:
        ------------
            Set value of gamma threshold for width
        ________________________________________________________________________

        Args:
        -----
        :param gamma: float, int, None
            Value of gamma threshold for width
        :type gamma: float, int, None
        """
        if gamma is None:
            self.gamma = 10
        else:
            self.gamma = gamma

        self.gamma_w_m = self.gamma * self.w_m
        return

    def set_data_source(self, data_source: str):
        """
        Description:
        ------------
            Set data source to extract data from.
        ________________________________________________________________________

        Args:
        -----
        :param data_source: str
            Set data source to extract data from. Options are:
                'original': Original data
                'resample': Resampled data
                'smooth': Smoothed data
        :type data_source: str
        """
        if data_source.lower() not in ("original", "resample", "smooth"):
            self.logger.warning(
                "Data source not recognized, " "setting to resample"
            )
            self.data_source = "resample"
        else:
            self.data_source = data_source.lower()
        return

    # --------------------
    # Report Info
    # --------------------
    def extract_data_to_save(self):
        """
        Description:
        ------------
            Extract data to save.
        """
        attributes = inspect.getmembers(
            self, lambda a: not (inspect.isroutine(a))
        )

        no_core_attributes = [
            a
            for a in attributes
            if not (a[0].startswith("__") and a[0].endswith("__"))
            and a[0] != "logger"
            and not (a[0].startswith("_"))
        ]

        dict_general = {
            a[0]: a[1] for a in no_core_attributes if a[1] is not None
        }
        dict_to_save = {}
        for a in dict_general.items():
            a = list(a)
            # Catch deprication warnings
            if isinstance(a[1], DeprecationWarning):
                continue
            if a[1] is None:
                continue
            if isinstance(a[1], pd.DataFrame):
                a = list(a)
                if len(a[1]) == 0:
                    continue
                else:
                    a[1] = {i: a[1][i].values for i in a[1].columns}
            if isinstance(a[1], (list, np.ndarray, dict)):
                if len(a[1]) == 0:
                    continue
            if isinstance(a[1], RiverTreeScales):
                a[1] = a[1].trees
            if isinstance(a[1], np.int64):
                dict_to_save[a[0]] = int(a[1])
            elif isinstance(a[1], np.float64):
                dict_to_save[a[0]] = float(a[1])
            else:
                dict_to_save[a[0]] = a[1]

        return dict_to_save

    # --------------------
    # Core Methods
    # --------------------
    def _extract_data_source(self, give_add_data=False):
        if self.data_source == "original":
            x = self.x_o
            y = self.y_o
            s = self.s_o
            additional_data = {
                "z": self.z_o,
                "comid": self.comid_o,
                "so": self.so_o,
                "da_sqkm": self.da_sqkm_o,
                "w_m": self.w_m_o,
                "within_waterbody": self.within_waterbody_o,
            }
        elif self.data_source == "resample" or self.data_source == "spline":
            x = self.x
            y = self.y
            s = self.s
            additional_data = {
                "z": self.z,
                "comid": self.comid,
                "so": self.so,
                "da_sqkm": self.da_sqkm,
                "w_m": self.w_m,
                "within_waterbody": self.within_waterbody,
            }
        elif self.data_source == "smooth":
            x = self.x_smooth
            y = self.y_smooth
            s = self.s_smooth
            additional_data = {
                "z": self.z,
                "comid": self.comid,
                "so": self.so,
                "da_sqkm": self.da_sqkm,
                "w_m": self.w_m,
            }
        else:
            raise ValueError(
                "Data source not recognized, "
                'set it to "original", "resample" or "spline"'
            )

        if give_add_data:
            return x, y, s, additional_data
        else:
            return x, y, s

    def set_planimetry_derivatives(self, planimetry_derivatives):
        """
        Description:
        ------------
            Set planimetry derivatives
        ________________________________________________________________________

        Args:
        -----
        :param planimetry_derivatives: dict,
            Dictionary with the planimetry derivatives, the keys are:
                'dxds': derivative of x with respect to s
                'dyds': derivative of y with respect to s
                'd2xds2': second derivative of x with respect to s
                'd2yds2': second derivative of y with respect to s
        :type planimetry_derivatives: dict
        """
        self.planimetry_derivatives = planimetry_derivatives
        return

    def set_splines(self, splines):
        """
        Description:
        ------------
            Set splines of the x and y coordinates of the planimetry
        ________________________________________________________________________

        Args:
        -----
        :param planimetry_derivatives: dict,
            Dictionary with the planimetry derivatives, the keys are:
                'dxds': derivative of x with respect to s
                'dyds': derivative of y with respect to s
                'd2xds2': second derivative of x with respect to s
                'd2yds2': second derivative of y with respect to s
        :type planimetry_derivatives: dict
        """
        self.splines = splines
        return

    def eval_splines(self, s_value):
        """
        Description:
        ------------
            evaluate splines of the x and y coordinates of the planimetry.
        ________________________________________________________________________

        Args:
        -----
        :param s_value: float or array-like,
            Value of s to evaluate the splines.
        :type s_value: float or array-like
        """
        s_v = copy.deepcopy(s_value)
        # The splines where fitted to the original coordinates
        x_spl = self.splines["x_spl"]
        y_spl = self.splines["y_spl"]
        if self.scale_by_width and not (np.isnan(self.w_m_gm)):
            s_v *= self.w_m_gm
        x = x_spl(s_v)
        y = y_spl(s_v)
        if self.scale_by_width and not (np.isnan(self.w_m_gm)):
            x /= self.w_m_gm
            y /= self.w_m_gm
        return x, y

    # --------------------
    # Modify River
    # --------------------
    def scale_coordinates(self, value: float):
        """
        Description:
        ------------
            Scale coordinates by a given value
        ________________________________________________________________________

        Args:
        -----
        :param value: float,
            Value to scale the coordinates.
        :type value: float
        """
        self.scale = value
        self.s /= self.scale
        self.x /= self.scale
        self.y /= self.scale
        self.z /= self.scale
        return

    def translate_xy_start(self):
        """
        Description:
        ------------
            Translate coordinates to xy start
        """
        self.translate = True
        self.x -= self.xy_start[0]
        self.y -= self.xy_start[1]
        return

    def calculate_spline(
        self, function=RF.fit_splines_complete, *args, **kwargs
    ):
        """
        Description:
        ------------
            Calculate spline of the river
        ________________________________________________________________________

        Args:
        -----
        :param function: function,
            Function to use for calculating the spline. Default is
            `RiverFunctions.fit_splines_complete`.
        :type function: function
        :param args: list,
            List of arguments to pass to the function.
        :type args: list
        :param kwargs: dict,
            Dictionary of arguments to pass to the function.
        :type kwargs: dict
        """
        data_pd = {
            "s": self.s_o,
            "x": self.x_o,
            "y": self.y_o,
            "z": self.z_o,
            "so": self.so_o,
            "comid": self.comid_o,
            "da_sqkm": self.da_sqkm_o,
            "w_m": self.w_m_o,
        }
        data_pd = copy.deepcopy(data_pd)
        # Add within_waterbody from comid values
        # get indices from unique
        comid_u, c_index = np.unique(self.comid_o, return_index=True)
        within_waterbody = {
            self.comid_o[c_i]: self.within_waterbody_o[c_i] for c_i in c_index
        }

        data = function(data_pd, *args, **kwargs)

        self.x = data["x_poly"]
        self.y = data["y_poly"]
        self.s = data["s_poly"]
        self.z = data["z_poly"]
        self.so = data["so_poly"]
        self.comid = data["comid_poly"]
        self.comid = np.array([str(int(c)) for c in self.comid])
        self.da_sqkm = data["da_sqkm_poly"]
        self.w_m = data["w_m_poly"]
        self.within_waterbody = np.zeros(len(self.x))

        if np.sum(comid_u == "") == 0:
            for c_u in comid_u:
                self.within_waterbody[self.comid == c_u] = within_waterbody[c_u]
        # Include planimetry derivatives
        self.set_planimetry_derivatives(data["derivatives"])
        self.set_splines(data["splines"])
        return

    def calculate_smooth(
        self,
        poly_order: int = 2,
        savgol_window: int = 3,
        gaussian_window: int = 1,
    ):
        """
        Description:
        ------------
            Smooth the planimetry of the river.
        ________________________________________________________________________

        Args:
        -----
        :param poly_order: int,
            Order of the polynomial to use for smoothing.
        :type poly_order: int
        :param savgol_window: int,
            Window size for the Savitzky-Golay filter.
        :type savgol_window: int
        :param gaussian_window: int,
            Window size for the Gaussian filter.
        :type gaussian_window: int
        """
        s_smooth, x_smooth, y_smooth = RF.smooth_data(
            self.x,
            self.y,
            self.s,
            poly_order=poly_order,
            savgol_window=savgol_window,
            gaussian_window=gaussian_window,
        )
        self.s_smooth = s_smooth
        self.x_smooth = x_smooth
        self.y_smooth = y_smooth
        self.ds_smooth = np.mean(np.diff(self.s_smooth))
        return

    def calculate_curvature(self, data_source: str = "resample"):
        """
        Description:
        ------------
            Calculate the curvature and the direction angle of the river.
        ________________________________________________________________________

        Args:
        -----
        :param data_source: str,
            Data source to use. Either 'original' or 'smooth'.
        """
        self.set_data_source(data_source)
        x, y, s, add_data = self._extract_data_source(give_add_data=True)
        # Calculate the curvature
        r, c, angle = RF.calculate_curvature(
            s, x, y, self.planimetry_derivatives
        )
        # convert angle to degrees
        angle = angle * 180 / np.pi
        # angle = RF.calculate_direction_angle(s, x, y, self.planimetry_derivatives)
        if self.scale_by_width and not (np.isnan(self.w_m_gm)):
            self.logger.info(" Scaling curvature by width")
            c *= self.w_m_gm
        self.r = r
        self.c = c
        self.angle = angle
        return

    def extract_cwt_tree(
        self, bound_to_poly: bool = False, cwt_kwargs: Union[dict, None] = None
    ):
        """
        Description:
        ------------
            Extracts meanders from the river network using CWT.
        _______________________________________________________________________

        Args:
        -----
        :param bounds_array_str: str,
            Bounds array string to use for the CWT.
            Either 'aggregated' or 'individual'.
        :param bound_to_poly: bool,
            If True, the CWT is bound to the polygon.
        :type bounds_array_str:
        :param cwt_kwargs: dict,
            Dictionary of CWT parameters. Look at function
            River.get_cwt() for parameters.
        :type cwt_kwargs: dict
        """

        # Error Management
        if len(self.c) == 0:
            self.logger.error('Please run "calculate_curvature" first.')
            raise ValueError("Please run 'calculate_curvature' first.")

        cwt_kwargs_original = {
            "pad": 1,
            "dj": 5e-2,
            "s0": -1,
            "j1": -1,
            "mother": "DOG",
            "m": 2,
        }
        if cwt_kwargs is None:
            cwt_kwargs = {}

        cwt_kwargs_original.update(cwt_kwargs)
        # ============================
        # Continuous Wavelet Transform
        # ============================
        self.logger.info("  Running CWT...")
        self.get_cwt_curvature(**cwt_kwargs_original)
        # ===================
        # Get the scale tree
        # ===================
        self.logger.info("  Getting the scale tree...")
        self.extract_tree()

        # =======================
        # Detect meanders on CWT
        # =======================
        self.logger.info("  Finding peaks in CWT...")
        self.find_peaks_in_poly()
        self.logger.info("  Detecting meanders...")
        self.detect_meanders_from_cwt()
        self.logger.info("  Projecting tree in planimetry...")
        self.get_tree_center_points_in_planimetry()
        # =======================
        # update scale_tree
        # =======================
        self.update_tree_scales(conn_source="ml_tree")
        return

    def get_cwt_curvature(
        self,
        pad: float = 1,
        dj: float = 5e-2,
        s0: float = -1,
        j1: float = -1,
        mother: str = "DOG",
        m: int = 2,
        sigtest=0,
        lag1=0,
        siglvl=0.95,
        dof=None,
    ):
        """
        Description:
        ------------
            Calculate the continuous wavelet transform (CWT) of the curvature.
        ________________________________________________________________________

        Args:
        -----
        :param pad: int,
            Add padding of zeros to the curvature.
        :param dj: float,
            Spacing between discrete scales.
        :param s0: float,
            Smallest scale of the wavelet. Default 2*ds.
        :param j1: float,
            Number of scales minus one. Default 1/dj.
        :param mother: str,
            Mother wavelet. Default 'DOG'.
        :param m: int,
            Parameter of the mother wavelet. Default 2.
        :return:
        """
        # Get curvature information
        c = self.c
        s_curvature = self.s
        # Calculate ds
        ds = np.diff(s_curvature)[0]
        # Calculate CWT
        wave, period, scales, power, gws, peak_period, sawp, coi, parameters = (
            WTFunc.calculate_cwt(c, ds, pad, dj, s0, j1, mother, m)
        )

        signif, sig95 = WTFunc.find_wave_significance(
            c, ds, scales, sigtest=0, lag1=0, siglvl=0.95
        )

        # Values of power where sig95 > 1 are significant
        sig95 = power / sig95
        power_sig95 = copy.deepcopy(power)
        power_sig95[sig95 <= 1] = 0
        wave_sig95 = copy.deepcopy(wave)
        wave_sig95[sig95 <= 1] = 0

        # Remove values within the cone of influence (COI)
        for i in range(len(coi)):
            cond = period > coi[i]
            power_sig95[cond, i] = 0
            wave_sig95[cond, i] = 0

        # Recalculate GWS and SAWP
        gws_sig95, peaks = cwt_func.calculate_global_wavelet_spectrum(
            wave_sig95
        )
        # peak_periods_sig95 = period[peaks]

        # Find SAWP (Spectral-Average Wave Period) using Zolezzi and Guneralp (2016)
        dj = parameters["dj"]
        c_delta = parameters["C_delta"]
        sawp_sig95 = cwt_func.calculate_scale_averaged_wavelet_power(
            wave_sig95, scales, ds, dj, c_delta
        )

        # Store data
        self.cwt_parameters_c = parameters
        self.cwt_wave_c = wave
        self.cwt_power_c = power
        self.cwt_wavelength_c = period
        self.cwt_scales_c = scales
        self.cwt_coi_c = coi
        self.cwt_gws_c = gws
        self.cwt_gws_peak_wavelength_c = peak_period
        self.cwt_sawp_c = sawp
        self.cwt_signif_c = signif
        self.cwt_sig95_c = sig95

        self.cwt_power_c_sig = power_sig95
        self.cwt_gws_c_sig = gws_sig95
        self.cwt_sawp_c_sig = sawp_sig95
        return

    def get_cwt_angle(
        self,
        pad: float = 1,
        dj: float = 5e-2,
        s0: float = -1,
        j1: float = -1,
        mother: str = "MORLET",
        m: int = 2,
    ):
        """
        Description:
        ------------
            Calculate the continuous wavelet transform (CWT) of the direction
            angle.
        ________________________________________________________________________

        Args:
        -----
        :param pad: int,
            Add padding of zeros to the curvature.
        :param dj: float,
            Spacing between discrete scales.
        :param s0: float,
            Smallest scale of the wavelet. Default 2*ds.
        :param j1: float,
            Number of scales minus one. Default 1/dj.
        :param mother: str,
            Mother wavelet. Default 'MORLET'.
        :param m: int,
            Parameter of the mother wavelet. Default 2.
        :return:
        """
        # Get curvature information
        angle = copy.deepcopy(self.angle)
        s_curvature = self.s
        # Calculate ds
        ds = np.diff(s_curvature)[0]
        # Calculate CWT
        wave, period, scales, power, gws, peak_period, sawp, coi, parameters = (
            WTFunc.calculate_cwt(angle, ds, pad, dj, s0, j1, mother, m)
        )

        signif, sig95 = WTFunc.find_wave_significance(
            angle, ds, scales, sigtest=0, lag1=0, siglvl=0.95
        )

        # Values of power where sig95 > 1 are significant
        sig95 = power / sig95
        power_sig95 = copy.deepcopy(power)
        power_sig95[sig95 <= 1] = 0
        wave_sig95 = copy.deepcopy(wave)
        wave_sig95[sig95 <= 1] = 0
        # Remove values within the cone of influence (COI)
        for i in range(len(coi)):
            cond = period > coi[i]
            power_sig95[cond, i] = 0
            wave_sig95[cond, i] = 0

        # Recalculate GWS and SAWP
        gws_sig95, peaks = cwt_func.calculate_global_wavelet_spectrum(
            wave_sig95
        )
        # peak_periods_sig95 = period[peaks]

        # Find SAWP (Spectral-Average Wave Period) using Zolezzi and Guneralp (2016)
        dj = parameters["dj"]
        c_delta = parameters["C_delta"]
        sawp_sig95 = cwt_func.calculate_scale_averaged_wavelet_power(
            wave_sig95, scales, ds, dj, c_delta
        )

        # Store data
        self.cwt_parameters_angle = parameters
        self.cwt_wave_angle = wave
        self.cwt_power_angle = power
        self.cwt_wavelength_angle = period
        self.cwt_scales_angle = scales
        self.cwt_coi_angle = coi
        self.cwt_gws_angle = gws
        self.cwt_gws_peak_wavelength_angle = peak_period
        self.cwt_sawp_angle = sawp
        self.cwt_signif_angle = signif
        self.cwt_sig95_angle = sig95

        self.cwt_power_angle_sig = power_sig95
        self.cwt_gws_angle_sig = gws_sig95
        self.cwt_sawp_angle_sig = sawp_sig95
        return

    def extract_tree(self):
        """
        Description:
        ------------
            Extract the tree from the CWT of the curvature.
        ________________________________________________________________________
        """
        conn, regions, poly, zc_lines, zc_sign = WTFunc.scale_space_tree(
            self.cwt_wave_c.real
        )
        # Store data
        self.cwt_conn = conn
        self.cwt_regions = regions
        self.cwt_poly = poly
        self.cwt_zc_lines = zc_lines
        self.cwt_zc_sign = zc_sign
        return

    def find_peaks_in_poly(self, remove_inside_coi: bool = False):
        """
        Description:
        ------------
            Find the peaks in the polygon.
        ________________________________________________________________________

        Args:
        -----
        :param remove_inside_coi: bool,
            If True, remove peaks that are inside the cone of influence.
        :return:
        """
        # Find peaks in the polygon
        peak_pwr, peak_row, peak_col = WTFunc.find_peak_in_poly(
            self.cwt_poly, self.cwt_wave_c.real
        )

        conn = self.cwt_conn
        coi = self.cwt_coi_c
        scales = self.cwt_scales_c

        # Remove nodes that are inside the cone of influence
        if remove_inside_coi:
            conn_unique = copy.deepcopy(conn)
            conn_unique[conn_unique < 0] = np.nan
            conn_unique = conn_unique[~np.isnan(conn_unique)].astype(int)
            peak_row_c = peak_row[conn_unique]
            peak_col_c = peak_col[conn_unique]
            peak_pwr_c = peak_pwr[conn_unique]

            # Remove nans
            peak_row_c = peak_row_c[~np.isnan(peak_row_c)].astype(int)
            peak_col_c = peak_col_c[~np.isnan(peak_col_c)].astype(int)
            peak_pwr_c = peak_pwr_c[~np.isnan(peak_pwr_c)]
            scales_c = scales[peak_row_c]
            coi_c = coi[peak_col_c]
            idx_all = []
            for i_s, s_c in enumerate(scales_c):
                if s_c >= coi_c[i_s]:
                    idx_all.append(i_s)

            # Remove value from connectivity and peaks
            conn[conn_unique[idx_all]] = -1
            peak_col[conn_unique[idx_all]] = np.nan
            peak_row[conn_unique[idx_all]] = np.nan
            peak_pwr[conn_unique[idx_all]] = np.nan
            self.cwt_conn = conn

        # Store data
        self.cwt_peak_pwr = peak_pwr
        self.cwt_peak_row = peak_row
        self.cwt_peak_col = peak_col
        return

    def detect_meanders_from_cwt(self):
        """
        Description:
        ------------
            Detect meanders from the CWT of the curvature.
        ________________________________________________________________________
        """
        if len(self.cwt_peak_pwr) <= 5:
            self.logger.warning("Not to many points to extract meanders.")
            self.cwt_meander_id = np.array([np.nan])
            self.cwt_ml_tree = np.array([np.nan])
        else:
            # Detect meanders
            frm = np.where(np.isnan(self.cwt_peak_pwr))[0]
            conn_2 = WTFunc.remove_nodes(self.cwt_conn, frm)
            meander_id = WTFunc.detect_meanders(
                self.cwt_wave_c.real,
                conn_2,
                self.cwt_peak_row,
                self.cwt_peak_col,
            )
            conn_2 = np.array(conn_2)
            meander_id = np.array(meander_id)
            # Clean meanders
            ml_tree = WTFunc.clean_tree(conn_2, meander_id)
            # Store data
            self.cwt_meander_id = meander_id
            self.cwt_ml_tree = ml_tree
        return

    def get_tree_center_points_in_planimetry(self):
        """
        Description:
        ------------
            Get the center points of the meanders.
        ________________________________________________________________________
        """
        bound_to_poly = False
        # Exctract coordinates from data source
        x, y, s = self._extract_data_source()
        self.bound_to_poly = bound_to_poly
        ds = np.diff(self.s)[0]

        if len(self.cwt_peak_pwr) <= 5:
            self.logger.warning("Not to many points to extract meanders.")
            self.cwt_planimetry_coords = np.array([np.nan])
        else:
            x_c, y_c = WTFunc.get_centers(
                self.cwt_ml_tree,
                self.cwt_peak_row,
                self.cwt_peak_col,
                self.cwt_wavelength_c,
                ds,
                x,
                y,
                extract_all=False,
                bound_to_poly=bound_to_poly,
            )
            # Store data
            self.cwt_planimetry_coords = np.vstack((x_c, y_c)).T

        return

    def update_tree_scales(self, conn_source: str = "conn"):
        """
        Description:
        ------------
            Update the tree scales object.
        ________________________________________________________________________

        Args:
        -----
        :param conn_source: str,
            Connectivity source to use. Either 'conn' or 'ml_tree'.
        :return:
        """
        # Extract coordinates
        x, y, s = self._extract_data_source()
        s_curvature = self.s
        c = self.c
        ds = np.diff(s_curvature)[0]
        bound_to_poly = self.bound_to_poly

        if conn_source == "conn":
            conn = self.cwt_conn
        elif conn_source == "ml_tree":
            conn = self.cwt_ml_tree

        peak_row = self.cwt_peak_row
        peak_col = self.cwt_peak_col
        peak_pwr = self.cwt_peak_pwr
        wave = self.cwt_wave_c.real
        wavelength = self.cwt_wavelength_c
        scales = self.cwt_scales_c
        poly = self.cwt_poly

        if np.sum(conn >= 0) == 0:
            self.logger.warning("No tree found")
            # print('No tree found')
            self.tree_scales = None
        else:
            tree_scales = WTFunc.get_tree_scales_dict(
                conn,
                peak_row,
                peak_col,
                peak_pwr,
                wave,
                wavelength,
                scales,
                ds,
                x,
                y,
                s_curvature,
                poly,
                include_metrics=True,
                bound_to_poly=bound_to_poly,
            )
            # Create tree RiverTreeScales object
            trees = RiverTreeScales()
            trees.build_trees_from_tree_scales_dict(tree_scales)
            self.tree_scales = trees
            # Add s vector for inflection points
            # =============================
            # Add Information into nodes
            # =============================
            tree_ids = self.tree_scales.tree_ids
            for tree_id in tree_ids:
                s_curvature = self.s
                root_node = self.tree_scales[tree_id]
                nodes = [root_node] + list(root_node.descendants)
                for node in nodes:
                    # ----------------------------------
                    # Find maximum point in curvature
                    # ----------------------------------
                    idx_start = node.idx_planimetry_start
                    idx_end = node.idx_planimetry_end
                    c_m = c[idx_start : idx_end + 1]
                    s_m = s_curvature[idx_start : idx_end + 1]
                    node.c = c_m
                    node.s = s_m
                    idx = np.argmax(np.abs(c_m))
                    idx = idx_start + idx
                    node.idx_c_max = idx_start
                    # --------------------------------------
                    # Update distance of the maximum point
                    # --------------------------------------
                    s_c = s_curvature[idx]
                    node.s_c = s_c
                    # ----------------------------------
                    # Add distance of the middle point
                    # ----------------------------------
                    # s_c = node.s_c
                    # idx = np.argmin(np.abs(s_curvature - s_c))
                    # ----------------------------------
                    # Add width of the middle point
                    # ----------------------------------
                    node.w_m = self.w_m[idx]
                    # ----------------------------------
                    # Add drainage area of the middle point
                    # ----------------------------------
                    node.da_sqkm = self.da_sqkm[idx]
                    # ----------------------------------
                    # Calculate curvature and sign
                    # ----------------------------------
                    node.c_c = c[idx]
                    node.sign_c = np.sign(node.c_c)
                    # ----------------------------------
                    # Calculate theta and sign
                    # ----------------------------------
                    r_x = node.r_x
                    r_y = node.r_y
                    theta_r = np.arctan2(r_y, r_x)
                    sign_r = np.sign(theta_r)
                    node.theta_r = theta_r
                    node.sign_r = sign_r
                    # ----------------------------------
                    # Add COMID
                    # ----------------------------------
                    start_comid = self.start_comid
                    node.start_comid = start_comid
                    comid_c = self.comid[idx]
                    node.comid_c = comid_c
                    # ----------------------------------
                    # Stream order
                    # ----------------------------------
                    so_c = self.so[idx]
                    node.stream_order_c = so_c
                    # ----------------------------------
                    # Find direction node to parent flag
                    # ----------------------------------
                    # Correct x_c and y_c
                    x_c = node.x_c
                    y_c = node.y_c
                    # x_c = x[idx]
                    # y_c = y[idx]
                    # node.x_c = x_c
                    # node.y_c = y_c
                    x_2 = node.x_2
                    y_2 = node.y_2
                    if node.parent is None:
                        node.direction_node_to_parent = 0
                    else:
                        x_c_parent = node.parent.x_c
                        y_c_parent = node.parent.y_c
                        dist_2_parent = np.sqrt(
                            (x_2 - x_c_parent) ** 2 + (y_2 - y_c_parent) ** 2
                        )
                        dist_c_parent = np.sqrt(
                            (x_c - x_c_parent) ** 2 + (y_c - y_c_parent) ** 2
                        )

                        # Find closest distance
                        if dist_2_parent < dist_c_parent:
                            direction_parent = -1
                        else:
                            direction_parent = 1

                        node.direction_node_to_parent = direction_parent

                    # ----------------------------------
                    # Find extended bounds
                    # ----------------------------------
                    node = RF.extend_node_bound(node, c)
                    # Calculate metrics
                    idx_start = node.idx_planimetry_extended_start
                    idx_end = node.idx_planimetry_extended_end
                    x_m = x[idx_start : idx_end + 1]
                    y_m = y[idx_start : idx_end + 1]
                    lambda_extended = RF.calculate_lambda(x_m, y_m)
                    l_extended = RF.calculate_l(x_m, y_m)
                    sn_extended = RF.calculate_sinuosity(
                        l_extended, lambda_extended
                    )
                    # Add values
                    node.lambda_extended = lambda_extended
                    node.l_extended = l_extended
                    node.sn_extended = sn_extended

                    # ----------------------------------
                    # Add s_reach
                    # ----------------------------------
                    node.s_reach = self.s[-1]

                    # ----------------------------------
                    # Add coordinates
                    # ----------------------------------
                    # Extract current distances
                    idx_start = node.idx_planimetry_start
                    idx_end = node.idx_planimetry_end
                    s_m = s[idx_start : idx_end + 1]
                    # Spline or Smooth
                    node.x = x[idx_start : idx_end + 1]
                    node.y = y[idx_start : idx_end + 1]
                    # Add original
                    idx_start_o = np.argmin(np.abs(s - s_m[0]))
                    idx_end_o = np.argmin(np.abs(s - s_m[-1]))
                    node.idx_planimetry_start_o = idx_start_o
                    node.idx_planimetry_end_o = idx_end_o
                    node.x_o = self.x_o[idx_start_o : idx_end_o + 1]
                    node.y_o = self.y_o[idx_start_o : idx_end_o + 1]
                    # ----------------------------------
                    # Find Inflection Points
                    # ----------------------------------
                    # plt.figure(figsize=(10, 5))
                    # plt.plot(s_m, c[idx_start:idx_end + 1])
                    # plt.axhline(0, color='k', linestyle='--')
                    # plt.show()
                    idx_start = copy.deepcopy(node.idx_planimetry_start)
                    idx_end = copy.deepcopy(node.idx_planimetry_end)
                    middle = int((idx_end - idx_start) / 2)
                    i_r = 0
                    s_inf = []
                    # find initial inflection point
                    while len(s_inf) != 1:
                        if idx_start - i_r < 0:
                            start = 0
                        else:
                            start = idx_start - i_r
                        c_m = c[start : middle + idx_start + 1]
                        s_m = s[start : middle + idx_start + 1]
                        s_inf, c_inf, ind_l, ind_r = RF.get_inflection_points(
                            s_m, c_m
                        )
                        i_r += 1
                        if idx_start - i_r < 0 and len(s_inf) == 0:
                            s_inf = np.array([s_m[0]])
                            break
                        # if i_r > 100:
                        #     break
                        if len(s_inf) >= 1:
                            s_inf = np.array([s_inf[0]])

                    # Check new inflection point with extended bounds
                    # print(f'Old Start Inflection {node.idx_planimetry_start}')
                    # print(f'New Start Inflection {start + 1}')
                    # print(f'Extended Bounds Start {node.idx_planimetry_extended_start}')
                    if start + 1 >= node.idx_planimetry_extended_start:
                        node.idx_planimetry_start = start + 1
                        s_inf_left = s_inf[0]
                    else:
                        # print('Starting inflection point out of full meander bounds. '
                        #       'Correcting to CWT bounds.')
                        self.logger.warning(
                            "Starting Inflection point out of full meander bounds. "
                            "Correcting to CWT bounds."
                        )
                        s_inf_left = s_curvature[node.idx_planimetry_start]

                    s_inf = []
                    i_r = 0
                    # Find final inflection point
                    while len(s_inf) != 1:
                        c_m = c[middle + idx_start : idx_end + i_r + 1]
                        s_m = s[middle + idx_start : idx_end + i_r + 1]
                        s_inf, c_inf, ind_l, ind_r = RF.get_inflection_points(
                            s_m, c_m
                        )
                        i_r += 1
                        if idx_end + i_r > len(c) and len(s_inf) == 0:
                            s_inf = np.array([s_m[-1]])
                            break
                        # if i_r > 100:
                        #     break
                        if len(s_inf) >= 1:
                            s_inf = np.array([s_inf[-1]])

                    # Check new inflection point with extended bounds
                    # print(f'Old End Inflection {node.idx_planimetry_end}')
                    # print(f'New End Inflection {idx_end + i_r}')
                    # print(f'Extended Bounds End {node.idx_planimetry_extended_end}')
                    if idx_end + i_r <= node.idx_planimetry_extended_end:
                        node.idx_planimetry_end = idx_end + i_r
                        s_inf_right = s_inf[0]
                    else:
                        # print('Ending inflection point out of full meander bounds. '
                        #       'Correcting to CWT bounds.')
                        self.logger.warning(
                            "Ending inflection point out of full meander bounds. "
                            "Correcting to CWT bounds."
                        )
                        s_inf_right = s_curvature[node.idx_planimetry_end]
                    if node.idx_planimetry_end >= len(c):
                        node.idx_planimetry_end = len(c) - 1

                    # Arrange values
                    s_inf = np.array([s_inf_left, s_inf_right])

                    # Evaluate the splines in the inflection points
                    x_inf, y_inf = self.eval_splines(s_inf)
                    node.s_inf = np.array([s_inf[0], s_inf[-1]])
                    node.x_inf = x_inf
                    node.y_inf = y_inf
                    # The inflection point should always be zero
                    # from the function above
                    node.c_inf = np.array([0, 0])

                    # ----------------------------------
                    # Add coordinates extended
                    # ----------------------------------
                    # Extract current distances
                    idx_start = node.idx_planimetry_extended_start
                    idx_end = node.idx_planimetry_extended_end
                    s_m = s[idx_start : idx_end + 1]
                    # Spline or Smooth
                    node.x_extended = x[idx_start : idx_end + 1]
                    node.y_extended = y[idx_start : idx_end + 1]
                    # Add original
                    idx_start_o = np.argmin(np.abs(self.s_o - s_m[0]))
                    idx_end_o = np.argmin(np.abs(self.s_o - s_m[-1]))
                    node.idx_planimetry_extended_start_o = idx_start_o
                    node.idx_planimetry_extended_end_o = idx_end_o
                    node.x_extended_o = self.x_o[idx_start_o : idx_end_o + 1]
                    node.y_extended_o = self.y_o[idx_start_o : idx_end_o + 1]

                    # ----------------------------------
                    # Add within_waterbody
                    # ----------------------------------
                    within_waterbody = self.within_waterbody[
                        idx_start : idx_end + 1
                    ]
                    if np.sum(within_waterbody) > 0:
                        node.within_waterbody = 1
                    else:
                        node.within_waterbody = 0

                    # ----------------------------------
                    # Add bounds of the leaves
                    # ----------------------------------
                    leaves = node.leaves
                    left_bounds = []
                    rigth_bounds = []
                    for leaf in leaves:
                        left_bounds.append(leaf.idx_planimetry_start)
                        rigth_bounds.append(leaf.idx_planimetry_end)

                    node.idx_leaf_start = np.min(left_bounds)
                    node.idx_leaf_end = np.max(rigth_bounds)

                    # Extract global wavelet spectrum (GWS)
                    wave_node = wave[
                        :, node.idx_leaf_start : node.idx_leaf_end + 1
                    ]
                    n = wave_node.shape[1]
                    gws = np.sum(np.abs(wave_node) ** 2, axis=1) / n
                    # Find peaks in the GWS
                    peaks, _ = find_peaks(gws)
                    peak_wavelength = wavelength[peaks]
                    node.gws = gws
                    node.gws_peak_wavelength = peak_wavelength

            self.tree_scales.update_database()
            self.tree_scales_database = self.tree_scales.compile_database()
            self.update_tree_scales_meander_database()
        return

    def update_tree_scales_meander_database(self):
        # Check current meanders
        # Correct meanders
        tree_ids = self.tree_scales.tree_ids
        nodes = self.tree_scales.filter_nodes(tree_ids, "is_meander", 1)
        for tree_id in tree_ids:
            for node in nodes[tree_id]:
                ind_start = node.idx_planimetry_start
                ind_end = node.idx_planimetry_end
                within_waterbody = node.within_waterbody
                if ind_start == ind_end:
                    # remove meander from tree scales
                    node.is_meander = 0
                    self.logger.warning(
                        "Meander with only one point detected. " "Removing..."
                    )
                if ind_end - ind_start > len(self.x) * 0.8:
                    node.is_meander = 0
                    self.logger.warning(
                        "Large system detected. Error in the tree"
                        "construction. Removing..."
                    )
                if within_waterbody == 1:
                    node.is_meander = 0
                    self.logger.warning(
                        "Meander within waterbody detected. " "Removing..."
                    )
        # Update database
        self.tree_scales.update_database()
        self.tree_scales_database = self.tree_scales.compile_database()
        database = self.tree_scales_database
        database_meanders = database[database["is_meander"] == 1]
        # don't take into account in meander within_waterbodies
        database_meanders = database_meanders[
            database_meanders["within_waterbody"] == 0
        ]
        # don't take into account in meander with large radius of curvature
        characteristic_length = self.s[-1] * 0.1
        database_meanders = database_meanders[
            database_meanders["radius"] < characteristic_length
        ]
        # sort by s_c
        database_meanders = database_meanders.sort_values(by="s_c")
        # renumber meander_id
        database_meanders["meander_id"] = np.arange(len(database_meanders))
        # Add meander_id to tree_scales
        self.tree_scales_database_meanders = database_meanders
        return

    def _update_tree_scales_leaf_bounds(self):
        tree_ids = self.tree_scales.tree_ids
        for tree_id in tree_ids:
            root_node = self.tree_scales[tree_id]
            nodes = [root_node] + list(root_node.descendants)
            for node in nodes:
                # ----------------------------------
                # Add bounds of the leaves
                # ----------------------------------
                leaves = node.leaves
                left_bounds = []
                rigth_bounds = []
                for leaf in leaves:
                    if leaf.is_meander == 1:
                        left_bounds.append(leaf.idx_planimetry_start)
                        rigth_bounds.append(leaf.idx_planimetry_end)

                if len(left_bounds) > 0:
                    node.idx_leaf_start = np.min(left_bounds)
                if len(rigth_bounds) > 0:
                    node.idx_leaf_end = np.max(rigth_bounds)
        return

    def add_meanders_from_tree_scales(
        self, bounds_array_str="inflection", overwrite=False, clip="no"
    ):
        """
        Description:
        ------------
            Add meanders from the database
        ________________________________________________________________________

        Args:
        -----
        :param bounds_array_str: str, Default 'inflection'.
            This parameter serves as a flag to indicate if the user wants to
            save the full-meander extent ('extended') or only the extent of the
            inflection points ('inflection').
        :param overwrite: bool, Default False.
            If True, overwrite the current meanders. Default False.
        :param clip: str, Default 'no'.
            Clip meander bounds to other meander bounds to have a train of meanders.
            Only works if bounds_array_str='inflection'. The possible options are:
            'no': will not clip the meanders bounds.
            'downstream': will clip the downstream begining meander bound to the upstream ending
                          meander bound.
            'upstream': will clip the upstream ending meander bound to the downstream begining
                        meander bound.
        :return:
        """
        if overwrite:
            self.id_meanders = []
            self.meanders = {}
            self.database = {i: [] for i in self._database_vars}
            self.database = pd.DataFrame.from_dict(self.database)
        # Error management
        if self.tree_scales_database is None:
            if self.tree_scales is None:
                raise AttributeError(
                    "Tree scales not calculated. Run "
                    "River.update_tree_scales()"
                )

        if bounds_array_str == "extended":
            ext = "_extended"
            inflection_flag = False
        elif bounds_array_str == "inflection":
            ext = ""
            inflection_flag = True
        else:
            raise ValueError(
                'bounds_array_str must be "extended" or ' '"inflection"'
            )

        # Clip to meanders
        self.update_tree_scales_meander_database()
        database_meanders = copy.deepcopy(self.tree_scales_database_meanders)

        if len(self.id_meanders) == 0:
            meander_id = 0
        else:
            meander_id = np.max(self.id_meanders) + 1
        meander_id_prev = copy.deepcopy(meander_id)
        # Loop through meanders
        for i in range(len(database_meanders)):
            ind_start = database_meanders[f"idx_planimetry{ext}_start"].iloc[i]
            ind_end = database_meanders[f"idx_planimetry{ext}_end"].iloc[i]
            sk = database_meanders["sk"].iloc[i]
            fl = database_meanders["fl"].iloc[i]
            tree_id = database_meanders["tree_id"].iloc[i]
            x_inf = database_meanders["x_inf"].iloc[i]
            y_inf = database_meanders["y_inf"].iloc[i]
            s_inf = database_meanders["s_inf"].iloc[i]
            wavelength = database_meanders["wavelength_c"].iloc[i]
            if inflection_flag:
                if clip.lower() == "downstream":
                    if meander_id_prev != meander_id:
                        ind_start = np.min([ind_start, ind_end_prev])
                        if ind_start == ind_end_prev:
                            x_inf_1 = database_meanders["x_inf"].iloc[i - 1][1]
                            y_inf_1 = database_meanders["y_inf"].iloc[i - 1][1]
                            s_inf_1 = database_meanders["s_inf"].iloc[i - 1][1]
                            x_inf = np.array([x_inf_1, x_inf[1]])
                            y_inf = np.array([y_inf_1, y_inf[1]])
                            s_inf = np.array([s_inf_1, s_inf[1]])

                elif clip.lower() == "upstream":
                    if i < len(database_meanders) - 1:
                        ind_pos_start = database_meanders[
                            f"idx_planimetry{ext}_start"
                        ].iloc[i + 1]
                        ind_end = np.max([ind_end, ind_pos_start])
                        if ind_end == ind_pos_start:
                            x_inf_1 = database_meanders["x_inf"].iloc[i + 1][0]
                            y_inf_1 = database_meanders["y_inf"].iloc[i + 1][0]
                            s_inf_1 = database_meanders["s_inf"].iloc[i + 1][0]
                            x_inf = np.array([x_inf[0], x_inf_1])
                            y_inf = np.array([y_inf[0], y_inf_1])
                            s_inf = np.array([s_inf[0], s_inf_1])

            node_id = database_meanders["node_id"].iloc[i]
            node = self.tree_scales.filter_nodes([tree_id], "node_id", node_id)
            if ind_start == ind_end:
                # remove meander from tree scales
                node[tree_id][0].is_meander = 0
                self.logger.warning(
                    "Meander with only one point detected. " "Removing..."
                )
                continue
            if ind_end - ind_start > len(self.x) * 0.8:
                node[tree_id][0].is_meander = 0
                self.logger.warning(
                    "Large system detected. Error in the tree"
                    "construction. Removing..."
                )
                continue
            try:
                self.add_meander(
                    meander_id,
                    ind_start,
                    ind_end,
                    sk=sk,
                    fl=fl,
                    automatic_flag=1,
                    inflection_flag=inflection_flag,
                    tree_id=tree_id,
                    x_inf=x_inf,
                    y_inf=y_inf,
                    s_inf=s_inf,
                    wavelength=wavelength,
                )
            except SmallMeanderError:
                self.logger.warning("Small Meander detected. Removing...")
                continue

            meander_id_prev = copy.deepcopy(meander_id)
            meander_id += 1
            ind_end_prev = ind_end

    def prune_tree_by_gamma_width(self, gamma):
        """
        Description:
        ------------
            Prune meanders by gamma width. Meanders with gamma width smaller
            than w_m_factor * gamma width are removed.

            This function will change the key 'meander_in_level_root_leaf' in
            tree_scales. The key 'meander_in_level_root_leaf' contains the
            level where the meander is located in each branch.
        ________________________________________________________________________

        Args:
        -----
        :param gamma: float
            Factor to multiply gamma width. Meanders with gamma width smaller
            than w_m_factor * gamma width are removed.
        ________________________________________________________________________
        """

        self.set_gamma_width(gamma)

        if self.tree_scales is None:
            self.logger.error(
                "Tree scales not calculated. Run " "River.calculate_tree()"
            )
            raise ValueError(
                "Tree scales not calculated. Run " "River.calculate_tree()"
            )

        # Extract data
        tree_scales = self.tree_scales
        # Perform pruning
        tree_scales.prune(
            method="width",
            width_var="w_m",
            compare_var="wavelength_c",
            gamma=gamma,
        )
        # Update Tree Scales
        self.tree_scales = tree_scales
        self.update_tree_scales_meander_database()
        self._update_tree_scales_leaf_bounds()
        return

    def prune_tree_by_peak_power(self, factor=0.1):
        """
        Description:
        ------------
            Prune the tree scales by comparing the peak power of the parent
            to the ones of the children
        """
        if self.tree_scales is None:
            self.logger.error(
                "Tree scales not calculated. Run " "River.calculate_tree()"
            )
            raise ValueError(
                "Tree scales not calculated. Run " "River.calculate_tree()"
            )
        tree_scales = self.tree_scales
        characteristic_length = self.s[-1] * factor
        # perform pruning
        tree_scales.prune(
            method="peak_power",
            peak_power_var="peak_pwr",
            sign_var="direction_node_to_parent",
            characteristic_length=characteristic_length,
        )

        self.tree_scales = tree_scales
        self.update_tree_scales_meander_database()
        self._update_tree_scales_leaf_bounds()
        return

    def prune_tree_by_sinuosity(
        self,
        sinuosity_threshold=1.1,
        sinuosity_var="sn",
    ):
        """
        Description:
        ------------
            Prune the meander nodes that have a sinuosity lower than the
            sinuosity_threshold.
        ________________________________________________________________________

        Args:
        -----
        :param sinuosity_var: str,
            Sinuosity variable to use. Default 'sn'.
        :param sinuosity_threshold: float,
            Sinuosity threshold. Default 1.1.
        """
        if self.tree_scales is None:
            self.logger.error(
                "Tree scales not calculated. Run " "River.calculate_tree()"
            )
            raise ValueError(
                "Tree scales not calculated. Run " "River.calculate_tree()"
            )
        tree_scales = self.tree_scales
        tree_scales.prune(
            method="sinuosity",
            sinuosity_var=sinuosity_var,
            sinuosity_threshold=sinuosity_threshold,
        )

        self.tree_scales = tree_scales
        self.update_tree_scales_meander_database()
        self._update_tree_scales_leaf_bounds()
        return

    # -------------------------
    # Plotting
    # -------------------------
    def plot_tree_nodes(self, include_width=True, title=None, **kwargs):
        """
        Description:
        ------------
            Plot tree nodes.
        ________________________________________________________________________

        Args:
        -----
        :param include_width: bool,
            If True, include width in the plot. Default True.
        :param kwargs:
            Keyword arguments to pass to the plot function.
            Look in utilities.graphs.plot_tree_from_anytree for more
            information in these arguments.
        """
        x, y, s = self._extract_data_source()

        s_curvature = self.s
        scales = self.cwt_scales_c
        wavelength = self.cwt_wavelength_c
        power = np.log2(self.cwt_power_c)
        # wave = np.log2(self.cwt_wave**2)
        coi = self.cwt_coi_c
        tree_scales = self.tree_scales
        gws = self.cwt_gws_c
        peaks_gws = self.cwt_gws_peak_wavelength_c
        id_river = self.id_value
        scale_by_width = self.scale_by_width
        # peaks_min, min_s = RF.calculate_spectrum_cuts(s, self.cwt_wave)
        peaks_min, min_s = RF.calculate_spectrum_cuts(s, self.c)
        if include_width:
            gamma_width = self.gamma_w_m
        else:
            gamma_width = None

        graphs.plot_tree_from_anytree(
            x,
            y,
            s_curvature,
            wavelength,
            power,
            tree_scales,
            gws,
            peaks_gws,
            id_river,
            coi=coi,
            min_s=None,
            scale_by_width=scale_by_width,
            title=title,
            **kwargs,
        )

    def interactive_meander_characterization_plot(
        self, inflection_flag=False, mapbox_token=None, **kwargs
    ):
        """
        Description:
        ------------
            Interactive meander characterization. This function will only work
            in a Jupyter Notebook.

            In this function, the idea is to select full meanders that only
            has one lobe and one neck. The user can also pick to select the
            half-meanders based on the location of the inflection points by
            setting inflection_flag=True.
        ________________________________________________________________________

        """
        if self.scale_by_width and mapbox_token is not None:
            raise ValueError(
                "Due to coordinates issues the Interactive Meander"
                " Characterization is not implemented for"
                " scale_by_width=True"
            )
        clicked_points = []
        meander_ids = copy.deepcopy(self.id_meanders)
        x, y, z = self._extract_data_source()
        f = iplot.plot_interactive_river_plain(
            x,
            y,
            clicked_points=clicked_points,
            meander_ids=meander_ids,
            river_obj=self,
            inflection_flag=inflection_flag,
            mapbox_token=mapbox_token,
            **kwargs,
        )

        return f

    def plot_meander(self, meander_id, engine="matplotlib"):
        """
        Description:
        ------------
            Plot meander.
        ________________________________________________________________________
        """
        x, y, s = self._extract_data_source()
        x_meander = self.meanders[meander_id].x
        y_meander = self.meanders[meander_id].y
        if engine == "matplotlib":
            graphs.plot_meander_matplotlib(x, y, x_meander, y_meander)
        return

    # -------------------------
    # Reporting and Saving
    # -------------------------
    def select_database(self, database="meander"):
        if database.lower() == "tree_scales":
            database = self.tree_scales_database_meanders
        elif database.lower() == "meander":
            database = self.database
        else:
            raise ValueError(
                "Database not recognized. Use either "
                '"tree_scales" or "meander"'
            )

        return database

    def create_meander_geopandas_dataframe(
        self,
        database="meander",
        geometry_columns=["x_o", "y_o"],
        shape_type="line",
        crs="esri:102003",
    ):
        """
        Description:
        ------------
            Create a geopandas dataframe from the meander dataframe
        ________________________________________________________________________

        Args:
        -----
        :param database: str,
            Database to extract the meander information. Default 'tree_scales'.
        :param geometry_columns: list,
            List of columns to use as geometry. Default ['x_o', 'y_o'].
        :param shape_type: str,
            Shape type. Default 'line'.
        :param crs: str,
            Coordinate reference system. Default 'esri:102003'.
        ________________________________________________________________________
        """
        # Extract database
        database = self.select_database(database)

        gdf = FM.create_geopandas_dataframe(
            database,
            geometry_columns=geometry_columns,
            shape_type=shape_type,
            crs=crs,
        )

        return gdf

    def save_meanders_database(
        self,
        path_output,
        file_name,
        database="meander",
        type_info="shapefile",
        geometry_columns=["x_o", "y_o"],
        shape_type="line",
        crs="esri:102003",
        **kwargs,
    ):
        """
        Description:
        ------------
            Save meander database.
        ________________________________________________________________________

        Args:
        -----
        :param path_output: str,
            Path to save the file.
        :param file_name: str,
            File name.
        :param database: str,
            Database to extract the meander information. Default 'meander'.
        :param type_info: str,
            Type of file to save. Default 'shapefile'.
        :param geometry_columns: list,
            List of columns to use as geometry. Default ['x_o', 'y_o'].
        :param shape_type: str,
            Shape type. Default 'line'.
        :param crs: str,
            Coordinate reference system. Default 'esri:102003'.
        ________________________________________________________________________
        """
        if type_info == "shapefile":
            # Convert to geopandas dataframe
            save_df = self.create_meander_geopandas_dataframe(
                database=database,
                geometry_columns=geometry_columns,
                shape_type=shape_type,
                crs=crs,
            )

            save_df_up = save_df[save_df["curvature_side"] == 1]
            save_df_down = save_df[save_df["curvature_side"] == -1]

            file_name_no_ext = "".join(file_name.split(".")[:-1])
            file_name_up = file_name_no_ext + "_up." + file_name.split(".")[-1]
            file_name_down = (
                file_name_no_ext + "_down." + file_name.split(".")[-1]
            )
            FM.save_data(
                save_df_up,
                path_output=path_output,
                file_name=file_name_up,
                **kwargs,
            )
            FM.save_data(
                save_df_down,
                path_output=path_output,
                file_name=file_name_down,
                **kwargs,
            )
        else:
            save_df = copy.deepcopy(self.select_database(database))
            save_df.reset_index(inplace=True)
            FM.save_data(
                save_df, path_output=path_output, file_name=file_name, **kwargs
            )
        return

    def save_river_coordinates(
        self,
        path_output,
        file_name,
        type_info="shapefile",
        geometry_columns=["x_o", "y_o"],
        shape_type="line",
        crs="esri:102003",
        **kwargs,
    ):
        """
        Description:
        ------------
            Save river object.
        ________________________________________________________________________

        Args:
        -----
        :param path_output: str,
            Path to save the file.
        :param file_name: str,
            File name.
        ________________________________________________________________________
        """
        # TODO: Incorporate scaling by width in the saving coordinates
        x, y, z, data = self._extract_data_source(give_add_data=True)
        s = data["s"]
        id_river = self.id_value
        uid = self.uid
        x_o = self.x_o
        y_o = self.y_o
        s_o = self.s_o
        so_o = self.so_o
        z_o = self.z_o
        comid_o = self.comid_o
        w_m_o = self.w_m_o
        da_sqkm_o = self.da_sqkm_o
        huc04 = self.huc04
        huc_n = self.huc_n
        start_comid = self.start_comid

        # create dataframe
        save_df = pd.DataFrame(
            {
                "id_river": id_river,
                "uid": uid,
                "x": x,
                "y": y,
                "z": z,
                "s": s,
                "x_o": x_o,
                "y_o": y_o,
                "s_o": s_o,
                "so_o": so_o,
                "z_o": z_o,
                "comid_o": comid_o,
                "w_m_o": w_m_o,
                "da_sqkm_o": da_sqkm_o,
                "huc04": huc04,
                "huc_n": huc_n,
                "start_comid": start_comid,
            }
        )

        if type_info == "shapefile":
            save_df = FM.create_geopandas_dataframe(
                save_df,
                geometry_columns=geometry_columns,
                shape_type=shape_type,
                crs=crs,
            )

        FM.save_data(
            save_df, path_output=path_output, file_name=file_name, **kwargs
        )
        return

    # -------------------------
    # Meanders
    # -------------------------
    def add_meander(
        self,
        id_meander,
        ind_start,
        ind_end,
        metrics=None,
        sk=np.nan,
        fl=np.nan,
        automatic_flag=0,
        inflection_flag=False,
        tree_id=-1,
        x_inf=None,
        y_inf=None,
        s_inf=None,
        wavelength=None,
    ):
        """
        Description:
        ------------
            Add meander to the meander list
        ________________________________________________________________________

        Args:
        -----
        :param id_meander: int or str,
            Id of the meander.
        :param ind_start: int,
            Start index.
        :param ind_end: int,
            End index.
        :param curvature_side: int,
            Side of the curvature. Default np.nan.
        :param metrics: dict,
            Dictionary of metrics. Default None.
        :param c: np.array,
            Curvature. Default None.
        :param s_curvature: np.array,
            Distance along the curvature. Default None.
        :param sk: np.array,
            Skewness. Default None.
        :param fl: np.array,
            Flatness. Default None.
        :param automatic_flag: int,
            flag to add if the meander comes from the automated detection.
        :param inflection_flag: bool,
            flag to add if the points given are the inflection points
            of the meander.
        :param tree_id: int,
            Id of the tree where the meander belongs. Default -1.
        :return:
        """
        # ---------------
        # Clip data
        # ---------------
        x_data, y_data, s_data, add_data = self._extract_data_source(
            give_add_data=True
        )
        z_data = add_data["z"]
        so_data = add_data["so"]
        comid_data = add_data["comid"]
        w_m_data = add_data["w_m"]

        s = s_data[ind_start : ind_end + 1]
        x = x_data[ind_start : ind_end + 1]
        y = y_data[ind_start : ind_end + 1]
        z = z_data[ind_start : ind_end + 1]
        so = so_data[ind_start : ind_end + 1]
        w_m = w_m_data[ind_start : ind_end + 1]
        w_m_gm = self.w_m_gm
        if len(x) < 3:
            raise SmallMeanderError("Meander too small for current resolution.")

        # ---------------------------
        # Include inflection points
        # ---------------------------
        # Extract curvature
        if self.c is not None:
            c = self.c[ind_start : ind_end + 1]
        else:
            c = None
        if x_inf is not None:
            cond_1 = x[0] != x_inf[0] and y[0] != y_inf[0]
            cond_2 = x[-1] != x_inf[-1] and y[-1] != y_inf[-1]
            cond_3 = np.sum(s == s_inf[0]) > 0 or np.sum(s == s_inf[-1]) > 0
            if cond_1 and cond_2:
                x = np.hstack([x_inf, x])
                y = np.hstack([y_inf, y])
                s = np.hstack([s_inf, s])
                z = np.hstack([z_data[ind_start - 1], z_data[ind_end], z])
                so = np.hstack([so_data[ind_start - 1], so_data[ind_end], so])
                w_m = np.hstack(
                    [w_m_data[ind_start - 1], w_m_data[ind_end], w_m]
                )
                if c is not None:
                    c = np.hstack([0, 0, c])
            elif cond_1:
                x = np.hstack([x_inf[0], x])
                y = np.hstack([y_inf[0], y])
                s = np.hstack([s_inf[0], s])
                z = np.hstack([z_data[ind_start - 1], z])
                so = np.hstack([so_data[ind_start - 1], so])
                w_m = np.hstack([w_m_data[ind_start - 1], w_m])
                if c is not None:
                    c = np.hstack([0, c])
            elif cond_2:
                x = np.hstack([x, x_inf[-1]])
                y = np.hstack([y, y_inf[-1]])
                s = np.hstack([s, s_inf[-1]])
                z = np.hstack([z, z_data[ind_end]])
                so = np.hstack([so, so_data[ind_end]])
                w_m = np.hstack([w_m, w_m_data[ind_end]])
                if c is not None:
                    c = np.hstack([c, 0])

            s_sort = np.argsort(s)
            s = s[s_sort]
            x = x[s_sort]
            y = y[s_sort]
            z = z[s_sort]
            so = so[s_sort]
            w_m = w_m[s_sort]
            if c is not None:
                c = c[s_sort]

            # Correct for equal while adding inflection points
            if cond_3:
                s_un = np.unique(s)
                i_sort = np.argsort(s_un)
                s = s_un[i_sort]
                x = x[i_sort]
                y = y[i_sort]
                z = z[i_sort]
                so = so[i_sort]
                w_m = w_m[i_sort]
                if c is not None:
                    c = c[i_sort]

            if inflection_flag:
                i_s_inf_st = np.where(s == s_inf[0])[0][0]
                i_s_inf_end = np.where(s == s_inf[-1])[0][0]
                s = s[i_s_inf_st : i_s_inf_end + 1]
                x = x[i_s_inf_st : i_s_inf_end + 1]
                y = y[i_s_inf_st : i_s_inf_end + 1]
                z = z[i_s_inf_st : i_s_inf_end + 1]
                so = so[i_s_inf_st : i_s_inf_end + 1]
                w_m = w_m[i_s_inf_st : i_s_inf_end + 1]
                if c is not None:
                    c = c[i_s_inf_st : i_s_inf_end + 1]
                dif_idx = i_s_inf_end - i_s_inf_st
                if dif_idx < 3:
                    raise SmallMeanderError(
                        "Meander too small for current resolution."
                    )

        # ------------------------
        # Extract original data
        # ------------------------
        x_st = x[0]
        y_st = y[0]
        x_end = x[-1]
        y_end = y[-1]
        coords_st = np.array([x_st, y_st]).T
        coords_end = np.array([x_end, y_end]).T
        if self.scale_by_width and self.w_m_o is not None:
            x_o_all = self.x_o / self.w_m_o
            y_o_all = self.y_o / self.w_m_o
        else:
            x_o_all = self.x_o
            y_o_all = self.y_o
        coords_o_all = np.array([x_o_all, y_o_all]).T
        # Find closest point
        idx_start_o = np.argmin(
            np.linalg.norm(coords_o_all - coords_st, axis=1)
        )
        idx_end_o = np.argmin(np.linalg.norm(coords_o_all - coords_end, axis=1))
        # idx_start_o = np.argmin(np.abs(self.s_o - s[0]))
        # idx_end_o = np.argmin(np.abs(self.s_o - s[-1]))
        x_o = self.x_o[idx_start_o : idx_end_o + 1]
        y_o = self.y_o[idx_start_o : idx_end_o + 1]

        # Find the most recurring comid
        # comid = np.median(comid_data[ind_start: ind_end + 1])
        comid_val, comid_counts = np.unique(
            comid_data[ind_start : ind_end + 1], return_counts=True
        )
        comid = comid_val[np.argmax(comid_counts)]
        # -------------------------
        # Add meander
        # -------------------------
        self.id_meanders.append(id_meander)
        if np.min(so) < 1:
            a = 2

        meander = Meander(
            id_meander,
            s,
            x,
            y,
            z,
            ind_start,
            ind_end,
            so=so,
            x_o=x_o,
            y_o=y_o,
            ind_start_o=idx_start_o,
            ind_end_o=idx_end_o,
            x_inf=x_inf,
            y_inf=y_inf,
            s_inf=s_inf,
            w_m=w_m,
            w_m_gm=w_m_gm,
            c=c,
            wavelength=wavelength,
            metrics=metrics,
            sk=sk,
            fl=fl,
            comid=comid,
            automatic_flag=automatic_flag,
            inflection_flag=inflection_flag,
            tree_id=tree_id,
            scale_by_width=self.scale_by_width,
        )
        if len(self._calc_vars) == 0:
            self._calc_vars = meander.calc_vars
        # if metrics is None:
        #     meander.perform_calculations()
        self.meanders[id_meander] = meander
        # -------------------------
        # Add metrics to database
        # -------------------------
        database = {}
        database["id"] = [f"{self.uid}_{id_meander}"]
        database["id_reach"] = [self.uid]
        database["huc04_n"] = [str(self.id_value)]
        database["huc_n"] = [self.huc_n]
        database["start_comid"] = [self.start_comid]
        database["comid"] = [comid]
        database["scale"] = [self.scale]
        database["x_start_river"] = [self.x_start]
        database["y_start_river"] = [self.y_start]
        database["translate"] = [self.translate]
        database["id_meander"] = [id_meander]
        database["idx_start"] = [ind_start]
        database["idx_end"] = [ind_end]
        database["x_start"] = [x[0]]
        database["x_end"] = [x[-1]]
        database["y_start"] = [y[0]]
        database["y_end"] = [y[-1]]
        database["s_start"] = [s[0]]
        database["s_end"] = [s[-1]]
        database["s_middle"] = [s[0] + (s[-1] - s[0]) / 2]
        database["idx_start_o"] = [idx_start_o]
        database["idx_end_o"] = [idx_end_o]
        database["x"] = [x]
        database["y"] = [y]
        database["x_o"] = [x_o]
        database["y_o"] = [y_o]
        database["automatic_flag"] = [automatic_flag]
        database["inflection_flag"] = [inflection_flag]
        database["tree_id"] = [tree_id]
        database["w_m_gm"] = [w_m_gm]
        for calc in self._calc_vars:
            database[calc] = [meander.data[calc]]
        database = pd.DataFrame.from_dict(database)
        database.set_index("id", inplace=True)
        if len(self.database) == 0:
            self.database = database
        else:
            self.database = pd.concat([self.database, database])
        return

    def report_meanders_metrics(self):
        if len(self.meanders) == 0:
            raise ValueError("There is no meanders")
        return self.database

    def remove_meander(self, id_meander):
        """
        Description:
        ------------
            Remove meander from the meander list
        ________________________________________________________________________

        Args:
        -----
        :param id_meander: int or str,
            Id of the meander.
        """
        try:
            index = self.id_meanders.index(id_meander)
        except ValueError:
            return
        self.id_meanders.pop(index)
        self.meanders.pop(id_meander, None)
        id_database = f"{self.uid}_{id_meander}"
        self.database.drop(id_database, inplace=True)
        return

    def calculate_reach_metrics(self):
        """
        Description:
        ------------
            Calculate the sinuosity metrics for the reach.
        ________________________________________________________________________
        """

        # Extract Information
        database = self.database
        x = self.x
        y = self.y
        s = self.s

        d = RF.calculate_l(x, y)
        self.total_sinuosity = s[-1] / d

        try:
            tree_ids = np.unique(database["tree_id"].values)
        except KeyError:
            self.logger.warning(
                "No tree_id in database. No metrics calculated."
            )
            self.metrics_reach = None
            return
        # ----------------------------
        # Metrics for indiviual trees
        # ----------------------------
        variables = [
            "tree_id",
            "D",
            "l_i",
            "Y_k",
            "X_j",
            "total_sinuosity",
            "half_meander_sinuosity",
            "full_meander_sinuosity",
            "residual_meander_sinuosity",
            "mean_half_meander_length",
        ]
        tree_ids = np.hstack((-1, tree_ids))
        self.metrics_reach = {v: [0 for i in tree_ids] for v in variables}
        # self.metrics_reach = {str(i): {} for i in tree_ids}
        for i_t, tree_id in enumerate(tree_ids):
            # ===================
            # Distance metrics
            # ===================
            if tree_id == -1:
                subset_database = database
            else:
                subset_database = database[database["tree_id"] == tree_id]
            self.metrics_reach["tree_id"][i_t] = tree_id
            # Calculate D
            x_st = subset_database["x_start"].values[0]
            y_st = subset_database["y_start"].values[0]
            x_end = subset_database["x_end"].values[-1]
            y_end = subset_database["y_end"].values[-1]
            # calculate distance between start and end
            x_st_end = np.array([x_st, x_end])
            y_st_end = np.array([y_st, y_end])
            d = RF.calculate_l(x_st_end, y_st_end)
            self.metrics_reach["D"][i_t] = d
            # Calcualte l_i
            l_i = subset_database["lambda_hm"].values
            self.metrics_reach["l_i"][i_t] = l_i
            # Calcualte Y_k
            y_k = subset_database["L_hm"].values
            self.metrics_reach["Y_k"][i_t] = y_k
            # Calculate X_j
            n = subset_database.shape[0]
            i_start = np.arange(0, n - 1, 2)
            # i_end = np.arange(1, n, 2)
            x_f_m = subset_database["x_start"].values[i_start]
            y_f_m = subset_database["y_start"].values[i_start]
            x_f_m = np.hstack((x_f_m, subset_database["x_end"].values[-1]))
            y_f_m = np.hstack((y_f_m, subset_database["y_end"].values[-1]))
            coordinates = np.array([x_f_m, y_f_m]).T
            s_diff = np.diff(coordinates, axis=0)
            x_j = np.sqrt((s_diff**2).sum(axis=1))
            self.metrics_reach["X_j"][i_t] = x_j

            # ===================
            # Sinuosity Metrics
            # ===================
            # Calculate total sinuosity
            self.metrics_reach["total_sinuosity"][i_t] = np.sum(l_i) / d

            # Calculate half-meander sinuosity
            self.metrics_reach["half_meander_sinuosity"][i_t] = np.sum(
                l_i
            ) / np.sum(y_k)

            # Calculate full-meander sinuosity
            self.metrics_reach["full_meander_sinuosity"][i_t] = np.sum(
                y_k
            ) / np.sum(x_j)

            # Calculate residual sinuosity
            self.metrics_reach["residual_meander_sinuosity"][i_t] = (
                np.sum(x_j) / d
            )
            # Mean Meander length
            self.metrics_reach["mean_half_meander_length"][i_t] = np.mean(l_i)
        return


class Meander:
    """
    This class is the basic form of a meander. From the coordinates and height
    it calculates the basic metrics to report them.

    ===================== ==============================================================
    Attribute             Description
    ===================== ==============================================================
    s                     Vector of distances of each point.
    x                     x-dir coordiantes.
    y                     y-dir coordiantes.
    z                     Height of each point.
    ind_start             Start index in River.
    ind_end               End index in River.
    w_m                   Width of the River.
    w_m_gm                Geometric mean width of the River.
    c                     Curvature of the meander.
    sk                    Skewness.
    fl                    Flatness.
    comid                 COMID of the meander.
    x_o                   x-dir coordinates of the original river.
    y_o                   y-dir coordinates of the original river.
    ind_start_o           Start index in original river.
    ind_end_o             End index in original river.
    so                    Stream order of each point.
    x_inf                 x-dir coordinates of the inflection points.
    y_inf                 y-dir coordinates of the inflection points.
    s_inf                 Distance of the inflection points.
    wavelength            Wavelength of the meander.
    metrics               Calculated metrics.
    calculations          Boolean to perform calculations.
    automatic_flag        Boolean to indicate if the meander was automatically detected.
    inflection_flag       Boolean to indicate if the meander has inflection points.
    tree_id               Tree ID of the meander.
    scale_by_width        Boolean to scale the meander by the width of the river.
    ===================== ==============================================================

    The following are the methods of the class.

    ========================= =====================================================
    Methods                    Description
    ========================= =====================================================
    get_inflection_points      Get the inflection points of the meander.
    add_skewness               Add the skewness of the meander.
    add_flatness               Add the flatness of the meander.
    calculate_lambda           Calculates the length of the meander.
    calculate_l                Calculates the horizontal length of the meander.
    calculate_sinuosity        Calculates meander sinuosity.
    calculate_j_x              Calculate horizontal slope.
    calculate_so               Calculate the mode of stream order.
    calculate_wavelength       Calculate the wavelength of the meander.
    calculate_radius           Calculate the radius of the meander.
    plot_meander               Plot the meander.
    add_curvature              Add the curvature of the meander.
    add_s_inf                  Add the s_inf of the meander.
    calculate_asymetry         Calculate the asymetry of the meander.
    calculate_amplitude        Calculate the amplitude of the meander.
    calculate_funneling_factor Calculate the funneling factor of the meander.
    perform_calculations       Perform all the calculations.
    get_metrics                Get metric dict.
    ========================== =====================================================
    """

    def __init__(
        self,
        id_meander,
        s,
        x,
        y,
        z,
        ind_start,
        ind_end,
        w_m=None,
        w_m_gm=None,
        c=None,
        sk=np.nan,
        fl=np.nan,
        comid=np.nan,
        x_o=None,
        y_o=None,
        ind_start_o=None,
        ind_end_o=None,
        so=None,
        x_inf=None,
        y_inf=None,
        s_inf=None,
        wavelength=None,
        metrics=None,
        calculations=True,
        automatic_flag=0,
        inflection_flag=False,
        tree_id=-1,
        scale_by_width=False,
    ):
        # ----------------
        # Attributes
        # ----------------
        self._calc_vars = CALC_VARS
        self.id_meander = id_meander
        self.s = s
        self.x = x
        self.y = y
        # self.z = z
        self.ind_start = ind_start
        self.ind_end = ind_end
        self.automatic_flag = automatic_flag
        self.tree_id = tree_id
        self.inflection_flag = inflection_flag
        self.wavelength = wavelength
        self.scale_by_width = scale_by_width

        self.x_o = x_o
        self.y_o = y_o
        self.ind_start_o = ind_start_o
        self.ind_end_o = ind_end_o

        # Calculate curvature
        if c is None:
            r, c, theta = RF.calculate_curvature(s, x, y)
        self.c = c

        if w_m is None:
            w_m = np.ones_like(x) * np.nan
        self.w_m = w_m
        self.w_m_gm = w_m_gm

        # Do curvature side on the smooth data
        if inflection_flag:
            self.s_inf = [s[0], s[-1]]
            self.ind_inf_st = 0
            self.ind_inf_end = len(x) - 1
            self.x_inf_st = x[0]
            self.x_inf_end = x[-1]
            self.y_inf_st = y[0]
            self.y_inf_end = y[-1]
            # Add other variables
            self.c_smooth = copy.deepcopy(c)
            self.x_smooth = copy.deepcopy(x)
            self.y_smooth = copy.deepcopy(y)
            self.s_smooth = copy.deepcopy(s)
            curvature_side = np.sign(self.c[len(self.c) // 2])
            self.curvature_side = curvature_side
        elif x_inf is not None:
            # Find point closest to inflection point
            self.x_inf_st = x_inf[0]
            self.y_inf_st = y_inf[0]
            self.x_inf_end = x_inf[-1]
            self.y_inf_end = y_inf[-1]
            self.s_inf = s_inf
            # Extract Indices
            self.ind_inf_st = np.argmin(
                np.linalg.norm(
                    np.array([x, y]).T
                    - np.array([self.x_inf_st, self.y_inf_st]),
                    axis=1,
                )
            )
            self.ind_inf_end = np.argmin(
                np.linalg.norm(
                    np.array([x, y]).T
                    - np.array([self.x_inf_end, self.y_inf_end]),
                    axis=1,
                )
            )
            # Include distance of the inflection points
            # self.s_inf = [s[self.ind_inf_st], s[self.ind_inf_end]]
            # Add other variables
            self.c_smooth = copy.deepcopy(c)
            self.x_smooth = copy.deepcopy(x)
            self.y_smooth = copy.deepcopy(y)
            self.s_smooth = copy.deepcopy(s)
            curvature_side = np.sign(self.c[len(self.c) // 2])
            self.curvature_side = curvature_side
        else:
            self.get_infection_points()
            curvature_side = np.sign(self.c_smooth[len(self.c_smooth) // 2])
            self.curvature_side = curvature_side

        if so is None:
            self.so = [np.nan for _ in self.x]
        else:
            self.so = so

        self.comid = comid
        c_inf = c[self.ind_inf_st : self.ind_inf_end + 1]
        self.argmax_c = self.ind_inf_st + np.argmax(np.abs(c_inf))
        self.s_c_max = s[self.argmax_c]
        self.c_max = c[self.argmax_c]
        self.x_c_max = x[self.argmax_c]
        self.y_c_max = y[self.argmax_c]

        # ----------------------
        # Perform Calculations
        # ----------------------
        # Other metrics
        self.sk = sk
        self.fl = fl
        self.data = {i: np.nan for i in self._calc_vars}
        # Include location of inflection points
        self.data["x_inf"] = [self.x_inf_st, self.x_inf_end]
        self.data["y_inf"] = [self.y_inf_st, self.y_inf_end]
        self.data["s_inf"] = [self.s_inf[0], self.s_inf[-1]]
        self.data["x_c_max"] = self.x_c_max
        self.data["y_c_max"] = self.y_c_max
        self.data["s_c_max"] = self.s_c_max
        self.data["c_max"] = self.c_max
        self.data["w_m_gm"] = self.w_m_gm
        if metrics is None:
            if calculations:
                self.perform_calculations()
        else:
            self.data = metrics

        return

    def get_infection_points(self):
        """
        Description:
        -------------
            Have an estimate of the inflection points of the meander. This
            function calculates the curvature of the meander and then smooths
            the planimetry until it obtains the two inflection.

            This is an approximation. A better way to obtained these values is
            to use the wavelet analysis in the automated detection.
        """
        len_x_inf = 0
        i_iter = 0
        x = self.x
        y = self.y
        s = self.s
        n_p = len(x)
        if n_p < 5:
            c_smooth = copy.deepcopy(self.c)
            x_smooth = x
            y_smooth = y
            s_smooth = s
            s_inf = [s[0], s[-1]]
            idx_inf_st = 0
            idx_inf_end = n_p - 1
            x_inf_st = x[0]
            x_inf_end = x[-1]
            y_inf_st = y[0]
            y_inf_end = y[-1]
        else:
            gaussian_window = int(np.ceil(n_p / 10)) * 2
            while len_x_inf != 2:
                s_smooth, x_smooth, y_smooth = RF.smooth_data(
                    x, y, s, gaussian_window=gaussian_window
                )
                s_smooth += s[0]
                _, c_smooth, _ = RF.calculate_curvature(
                    s_smooth, x_smooth, y_smooth
                )
                s_inf, c_inf, ind_l, ind_r = RF.get_inflection_points(
                    s_smooth, c_smooth
                )

                len_x_inf = len(s_inf)
                idx_inf_st = 0
                idx_inf_end = len(x) - 1
                if len_x_inf == 2:
                    idx_inf_st = np.argmin(np.abs(s - s_inf[0]))
                    idx_inf_end = np.argmin(np.abs(s - s_inf[-1]))
                elif len_x_inf == 4 or len_x_inf == 3:
                    # Extract indices
                    cond = (ind_l > 0.1 * n_p) & (n_p - ind_l < 0.1 * n_p)
                    ind_l_ext = ind_l[~cond]
                    if len(ind_l_ext) >= 2:
                        s_inf = s_inf[~cond]
                        # check difference between indices in the points
                        dif_inds = np.diff(ind_l)
                        # Check for differences higher than 10% of the total number of points
                        dif_cond = ~(dif_inds > 0.1 * n_p)
                        sum_cond = np.sum(dif_cond)
                        # If the points obtained are close enough, take the first and last
                        if sum_cond in (1, 2):
                            idx_inf_st = np.argmin(np.abs(s - s_inf[0]))
                            idx_inf_end = np.argmin(np.abs(s - s_inf[-1]))
                            s_inf = [s_inf[0], s_inf[-1]]
                            len_x_inf = 2

                x_inf_st = x[idx_inf_st]
                y_inf_st = y[idx_inf_st]
                x_inf_end = x[idx_inf_end]
                y_inf_end = y[idx_inf_end]
                if i_iter == 0:
                    gaussian_window = int(np.ceil(n_p / 10)) * 2
                elif gaussian_window > 0.6 * n_p:
                    gaussian_window = int(np.ceil(n_p * 0.6)) * 2
                else:
                    gaussian_window *= 2

                # ----------------
                # Test with plot
                # ----------------
                # f,ax = plt.subplots(2, 1)
                # ax[0].plot(self.x, self.y, 'k')
                # ax[0].plot(x_smooth, y_smooth, 'r--')
                # ax[0].plot(x_inf_st, y_inf_st, 'ro')
                # ax[0].plot(x_inf_end, y_inf_end, 'ro')
                # ax[0].set_aspect('equal', adjustable='box')
                # ax[0].set_title(f'n_p={n_p}, gaussian_window={gaussian_window}\n'
                #                 f'len_x_inf={len_x_inf}, i_iter={i_iter}')
                # ax[1].plot(self.s, self.c, 'k')
                # ax_2 = ax[1].twinx()
                # ax_2.plot(s_smooth, c_smooth, 'r--')
                # ax_2.set_ylim([-0.005, None])
                # ax_2.axhline(0, color='k', linestyle='--')
                # plt.show()

                if i_iter > 4:
                    s_inf = [s[0], s[-1]]
                    break
                i_iter += 1

        # Correct coordinates with new inflection points?
        #  -> the location of the points is different to the actual coordinates
        #  because of the smoothing
        self.c_smooth = c_smooth
        self.x_smooth = x_smooth
        self.y_smooth = y_smooth
        self.s_smooth = s_smooth
        self.s_inf = s_inf
        self.ind_inf_st = idx_inf_st
        self.ind_inf_end = idx_inf_end
        self.x_inf_st = x_inf_st
        self.x_inf_end = x_inf_end
        self.y_inf_st = y_inf_st
        self.y_inf_end = y_inf_end
        return

    def add_skewness(self):
        self.data["skewness"] = self.sk
        return

    def add_flatness(self):
        self.data["flatness"] = self.fl
        return

    def calculate_lambda(self):
        coords = np.vstack((self.x, self.y)).T
        s_calc = RF.get_reach_distances(coords)
        self.data["lambda_fm"] = s_calc[-1]
        # Extract coords from inflection points
        coords = np.vstack(
            (
                self.x[self.ind_inf_st : self.ind_inf_end + 1],
                self.y[self.ind_inf_st : self.ind_inf_end + 1],
            )
        ).T
        s_calc = RF.get_reach_distances(coords)
        self.data["lambda_hm"] = s_calc[-1]
        return

    def calculate_l(self):
        self.data["L_fm"] = np.sqrt(
            (self.x[-1] - self.x[0]) ** 2 + (self.y[-1] - self.y[0]) ** 2
        )
        # Calculate l between inflection points
        x_inf = self.x[self.ind_inf_st : self.ind_inf_end + 1]
        y_inf = self.y[self.ind_inf_st : self.ind_inf_end + 1]
        self.data["L_hm"] = np.sqrt(
            (x_inf[-1] - x_inf[0]) ** 2 + (y_inf[-1] - y_inf[0]) ** 2
        )
        return

    def calculate_sinuosity(self):
        self.data["sigma_fm"] = self.data["lambda_fm"] / self.data["L_fm"]
        # Calculate sinuosity between inflection points
        self.data["sigma_hm"] = self.data["lambda_hm"] / self.data["L_hm"]
        return

    def calculate_j_x(self):
        dif_z = abs(self.z[-1] - self.z[0])
        self.data["dif_z"] = dif_z
        self.data["j_x"] = dif_z / self.data["l"]
        return

    def calculate_so(self):
        if np.all(np.isnan(self.so)):
            return
        try:
            self.data["so"] = st.mode(self.so, keepdims=True)[0][0]
        except TypeError:
            self.data["so"] = st.mode(self.so)[0][0]
        return

    def calculate_wavelength(self):
        """
        This is an estimate of the wavelength that from data turns out to be
        2 times the arc-length of the meander (lambda).
        """
        if self.wavelength is not None:
            self.data["lambda"] = copy.deepcopy(self.wavelength)
        else:
            self.data["lambda"] = self.data["lambda_fm"]
        return

    def calculate_radius(self):
        x = self.x[self.ind_inf_st : self.ind_inf_end + 1]
        y = self.y[self.ind_inf_st : self.ind_inf_end + 1]

        x_c, y_c, radius = RF.calculate_radius_of_curvature(
            x, y, self.data["lambda"] / 2
        )

        self.x_c = x_c
        self.y_c = y_c
        self.radius = radius
        self.data["R_hm"] = radius
        return

    def plot_meander(self, add_triangle=True):

        x = self.x
        y = self.y
        x_inf = self.x[self.ind_inf_st : self.ind_inf_end + 1]
        y_inf = self.y[self.ind_inf_st : self.ind_inf_end + 1]
        x_st = self.x_inf_st
        y_st = self.y_inf_st
        x_end = self.x_inf_end
        y_end = self.y_inf_end
        x_mid = x_inf[len(x_inf) // 2]
        y_mid = y_inf[len(y_inf) // 2]

        x_c = self.x_c
        y_c = self.y_c

        triangle_points = np.array(
            [[x_st, y_st], [x_mid, y_mid], [x_end, y_end], [x_st, y_st]]
        )
        f = plt.figure(figsize=(8, 8))
        plt.plot(x, y, "b")
        if add_triangle:
            plt.plot(triangle_points[:, 0], triangle_points[:, 1], "ro--")
            plt.plot(x_c, y_c, "bo")
            plt.plot([x_mid, x_c], [y_mid, y_c], "k-")
            # draw a circle
            circle = plt.Circle((x_c, y_c), self.radius, color="k", fill=False)
            plt.gca().add_artist(circle)
        plt.gca().set_aspect("equal")
        return f

    def add_curvature_side(self):
        self.data["curvature_side"] = self.curvature_side
        return

    def add_s_inf(self):
        self.data["s_inf"] = self.s_inf
        return

    def calculate_asymetry(self):
        a_h, lambda_h, lambda_u, lambda_d = RF.calculate_asymetry(
            self.x, self.y, self.c
        )
        # Store asymetry value
        self.data["a_fm"] = a_h
        self.data["lambda_fm,u"] = lambda_u
        self.data["lambda_fm,d"] = lambda_d

        # asymetry on half-meander
        x_inf = self.x[self.ind_inf_st : self.ind_inf_end + 1]
        y_inf = self.y[self.ind_inf_st : self.ind_inf_end + 1]
        c_inf = self.c[self.ind_inf_st : self.ind_inf_end + 1]
        a_h, lambda_h, lambda_u, lambda_d = RF.calculate_asymetry(
            x_inf, y_inf, c_inf
        )
        self.data["a_hm"] = a_h
        self.data["lambda_hm,u"] = lambda_u
        self.data["lambda_hm,d"] = lambda_d

        return

    def calculate_amplitude(self):
        """
        Calculate the amplitude of the meander by rotating the meander from
        the inflection points and calculating the distance between the minimum
        and maximum y values. Additionally, it calculates the dimensionless
        coordinate of the meander proposed by Lin and Limaye (2022).

        Equations:

        .. math:: A = \\max(y) - \\min(y)

        .. math:: x^* = \\frac{\\beta}{\\pi} - 0.5

        .. math:: A^* = \\frac{A}{w_m}

        example:

        .. code-block:: python

            x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            amplitude, x_star, center_point, coords_apex = RF.calculate_amplitude(x, y)

        References:

        Li, Y., & Limaye, A. B. (2022). Testing Predictions for Migration of
        Meandering Rivers: Fit for a Curvature-Based Model Depends on Streamwise
        Location and Timescale. Journal of Geophysical Research: Earth Surface,
        127(12), e2022JF006776. https://doi.org/10.1029/2022JF006776
        """
        # --------------------------
        # Extract Inflection points
        # --------------------------
        x_inf = self.x[self.ind_inf_st : self.ind_inf_end + 1]
        y_inf = self.y[self.ind_inf_st : self.ind_inf_end + 1]

        # ---------------------------------
        # Calculate amplitude Half-meander
        # ---------------------------------
        amplitude, x_star, center_point, coords_apex = RF.calculate_amplitude(
            x_inf, y_inf
        )
        self.data["A_hm"] = amplitude
        self.data["x_star_hm"] = x_star
        self.data["center_point_hm"] = center_point
        self.data["coords_apex_hm"] = coords_apex
        # ---------------------------------
        # Calculate amplitude Full-meander
        # ---------------------------------
        amplitude, x_star, center_point, coords_apex = RF.calculate_amplitude(
            self.x, self.y
        )
        self.data["A_fm"] = amplitude
        self.data["x_star_fm"] = x_star
        self.data["center_point_fm"] = center_point
        self.data["coords_apex_fm"] = coords_apex
        # ---------------------------------
        # Non-dimensional amplitude
        # ---------------------------------
        if not self.scale_by_width:
            self.data["A_star_fm"] = self.data["A_fm"] / self.data["w_m_gm"]
            self.data["A_star_hm"] = self.data["A_hm"] / self.data["w_m_gm"]
        else:
            print(
                "Coordiantes are already scaled by width, A_star is the same as A"
            )
            self.data["A_star_fm"] = self.data["A_fm"]
            self.data["A_star_hm"] = self.data["A_hm"]
        return

    def calculate_funneling_factor(self):
        """
        Calculate Funneling factor of the meander
        """
        try:
            results = RF.calculate_funneling_factor(
                self.x, self.y, self.s, self.ind_inf_st, self.ind_inf_end
            )
        except ValueError:
            results = {
                "FF": np.nan,
                "L_l": np.nan,
                "L_n": np.nan,
                "s_l": np.nan,
                "s_n": np.nan,
            }
            print(f"No Funneling Factor found for {self.id_meander}")
        self.data["FF"] = results["FF"]
        self.data["L_l"] = results["L_l"]
        self.data["L_n"] = results["L_n"]
        self.data["s_l"] = results["s_l"]
        self.data["s_n"] = results["s_n"]
        return

    def perform_calculations(self):
        functions = [
            self.calculate_lambda,
            self.calculate_l,
            self.calculate_wavelength,
            self.calculate_sinuosity,
            self.calculate_radius,
            # self.add_s_inf,
            self.calculate_so,
            self.add_flatness,
            self.add_skewness,
            self.add_curvature_side,
            self.calculate_asymetry,
            self.calculate_amplitude,
            self.calculate_funneling_factor,
            # self.calculate_j_x,
        ]

        for f in functions:
            f()

        return

    def get_metrics(self):
        return self.data
