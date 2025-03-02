# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#                               Last revised 2025-02-16
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
   This class extracts the complete reaches obtained with the model
"""
# -----------
# Libraries
# -----------
# System Management
import copy
import logging
import time

try:
    shell = get_ipython().__class__.__name__
    if shell == "ZMQInteractiveShell":
        from tqdm.notebook import tqdm  # Jupyter notebook or qtconsole
    elif shell == "TerminalInteractiveShell":
        from tqdm import tqdm  # Terminal running IPython
    else:
        from tqdm import tqdm  # Other type (console, script, etc.)
except NameError:
    from tqdm import tqdm  # Probably standard Python interpreter
from typing import Tuple
from typing import Union

# Data Management
import numpy as np
from scipy import interpolate
import pandas as pd

# MeanderCONUS Packages
from ..utilities import classExceptions as CE
from ..utilities import filesManagement as FM
from . import RiverFunctions as RF

# Determine if we're running in a Jupyter notebook

# ------------------
# Logging
# ------------------
# Set logger
logging.basicConfig(handlers=[logging.NullHandler()])


# ------------------
# Class
# ------------------
class CompleteReachExtraction:
    """
    This class obtained meander information from the NHD dataset.
    It works by loading the NHD geometry and using different methods
    to obtain meander information.

    The following are the available attributes

    ===================== =====================================================
    Attribute             Description
    ===================== =====================================================
    data                  Pandas dataframe with the NHD information.
    comid_id              String with the name of the comid column.
    logger                Logger object.
    ===================== =====================================================

    The following are the methods of the class.

    ============================= =====================================================
    Methods                       Description
    ============================= =====================================================
    load_coords                   Load coordinates from file.
    map_complete_network          Map comid network from the headwaters to the terminal
                                  nodes.
    map_complete_network_down_up  Map the entire database. It will do
                                  exploration from the terminal nodes to the
                                  headwaters. Saving the information in each reach.
    map_complete_network          Map comid network from the headwaters to the terminal
    map_complete_reach            Map individual comid network from starting comid to
    map_coordinates               Extract coordinates from the comid list.
    ============================= =====================================================
    """

    def __init__(self, data, comid_id="nhdplusid", logger=None, **kwargs):
        """
        Class constructor
        """
        # ------------------------
        # Logging
        # ------------------------
        if logger is None:
            self._logging = logging.getLogger(self.__class__.__name__)
            self._logging.setLevel(logging.DEBUG)
        else:
            self._logging = logger
            self._logging.info(f"Starting logger {self.__class__.__name__}")
        # ------------------------
        # Attribute management
        # ------------------------
        # Default data
        # Path management
        self.data_info = copy.deepcopy(data)
        # set headers to lower case
        self.data_info.columns = [x.lower() for x in self.data_info.columns]
        self.comid_id = comid_id.lower()
        self.pre_loaded_coords = False

        # Set index
        self.data_info.set_index(comid_id, inplace=True)
        self.huc_04 = np.unique(self.data_info["huc04"].values)
        self._save_format = "csv"
        self.__save_formats = FM.get_save_formats()

        # Change parameters
        valid_kwargs = ["save_format"]
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            k = f"_{k}"
            setattr(self, k, v)

        # --------------------
        # Data Management
        # --------------------
        # First create the base DataFrame with the essential columns
        try:
            self.linking_network = data.loc[
                :, ["nhdplusid", "startflag", "within_waterbody"]
            ].copy()
        except:
            self.logger.warning("No within_waterbody column found")
            self.linking_network = data.loc[
                :, ["nhdplusid", "startflag"]
            ].copy()

        # Set the index after copying to avoid modifying the original data
        self.linking_network.set_index("nhdplusid", inplace=True)

        # Create a dictionary of column definitions with their data types
        column_definitions = {
            "extracted_comid": pd.Series(dtype=np.int32),
            "linking_comid": pd.Series(dtype=str),
            "huc12": pd.Series(dtype=str),
            "huc10": pd.Series(dtype=str),
            "huc08": pd.Series(dtype=str),
            "huc06": pd.Series(dtype=str),
            "huc04": pd.Series(dtype=str),
            "huc02": pd.Series(dtype=str),
            "huc_n": pd.Series(dtype=np.int32),
            "n_tributaries": pd.Series(dtype=np.int32),
            "xm_m": pd.Series(dtype=np.float64),
            "ym_m": pd.Series(dtype=np.float64),
            "xup_m": pd.Series(dtype=np.float64),
            "yup_m": pd.Series(dtype=np.float64),
            "xdown_m": pd.Series(dtype=np.float64),
            "ydown_m": pd.Series(dtype=np.float64),
        }

        # Initialize all columns with proper data types and default values
        for col_name, dtype_series in column_definitions.items():
            if dtype_series.dtype == np.int32:
                self.linking_network[col_name] = pd.Series(
                    0, index=self.linking_network.index, dtype=np.int32
                )
            elif dtype_series.dtype == np.float64:
                self.linking_network[col_name] = pd.Series(
                    0.0, index=self.linking_network.index, dtype=np.float64
                )
            else:  # string type
                self.linking_network[col_name] = pd.Series(
                    "0", index=self.linking_network.index, dtype=str
                )

        data_hw = self.data_info[self.data_info["startflag"] == 1]
        start_comids = data_hw.index.values
        self.comid_network = {}
        self.extracted_comids = []

        # ------------
        # NHD Data
        # ------------
        self.coords_all = None
        return

    # --------------------------
    # Get functions
    # --------------------------
    @property
    def save_format(self):
        """save format of the files"""
        return self._save_format

    @property
    def logger(self):
        """logger for debbuging"""
        return self._logging

    # --------------------------
    # Set functions
    # --------------------------
    @save_format.setter
    def save_format(self, save_format):
        """set save format of the files"""
        if save_format not in self.__save_formats:
            self.logger.error(
                f"format '{save_format}' not implemented. "
                f"Use any of these formats "
                f"{self.__save_formats}"
            )
            raise CE.FormatError(
                f"format '{save_format}' not implemented. "
                f"Use formats 'p', 'mat', 'json', 'txt', or"
                f"'csv'"
            )
        else:
            self.logger.info(f"Setting save_format to '{save_format}'")
            self._save_format = save_format

    # --------------------------
    # Core functions
    # --------------------------
    def load_coords(self, path_coords: str) -> None:
        """
        DESCRIPTION:
        ------------
            Load coordinates from a file.
        ________________________________________________________________________

        Args:
        ------------
            :param path_coords: str
                Path to the coordinates file.
        """
        self.logger.info(f"Loading coordinates from {path_coords}")
        self.coords_all = FM.load_data(path_coords)
        self.pre_loaded_coords = True
        return

    def map_complete_network(
        self,
        start_comids: Union[list, np.ndarray, None] = None,
        huc_number: int = 4,
        max_num_comids: Union[int, None] = None,
        cut_comid_number: int = 3,
        do_not_overlap: bool = True,
    ) -> None:
        """
        DESCRIPTION:
        ------------
            Map comid network from the headwaters to the terminal nodes.
        ________________________________________________________________________

        Args:
        ------------
            :param start_comids: list, np.ndarray, Default None
                List of comids to be extracted.
            :param huc_number: int, Default 4
                HUC number to be extracted.
            :param max_num_comids: int, Default None
                Maximum number of comids to be extracted.
            :param cut_comid_number: int,
                Minimum number of comids contained in a reach to be extracted
            :param do_not_overlap: bool, Default False
                If True, the comids will not overlap.
        """
        # TODO: Include map newtwork as a function to huc level
        if start_comids is None:
            # Extract headwaters
            data_hw = self.data_info[self.data_info["startflag"] == 1]
            # data_hw = data_hw.sort_values(by='totdasqkm')
            data_hw = data_hw.sort_values(by="dnhydroseq", ascending=False)
            start_comids = np.array(data_hw.index)
        if isinstance(start_comids, str) or isinstance(start_comids, float):
            start_comids = [start_comids]

        if max_num_comids is None:
            max_num_comids = len(start_comids)

        # Look for the ones that have been not extracted
        linking_start = self.linking_network[
            self.linking_network["startflag"] == 1
        ]

        start_comids = start_comids[linking_start["extracted_comid"] == 0]
        # Loop over the comids
        self.extracted_comids = [0] * max_num_comids
        self.time = [0] * max_num_comids
        i_data = 0
        i_extracted = 0
        while i_data <= len(start_comids):
            time_1 = time.time()
            if i_extracted >= max_num_comids:
                break
            start_comid = start_comids[i_data]
            self.logger.info(f"Extracting comid {start_comid}")
            comid_network, _ = self.map_complete_reach(
                start_comid, huc_number, do_not_overlap=do_not_overlap
            )
            # Check if the comid_network is less than the cut_comid_number
            if len(comid_network) <= cut_comid_number:
                i_data += 1
                self.linking_network.loc[start_comid, "extracted_comid"] = -1
                continue
            self.comid_network[str(comid_network[0])] = comid_network
            self.extracted_comids[i_extracted] = start_comid
            self.linking_network.loc[start_comid, "extracted_comid"] = 1
            self.time[i_extracted] = (time.time() - time_1) / 60
            self.logger.info(f"Time elapsed: {self.time[i_extracted]} min")
            i_data += 1
            i_extracted += 1

        lengths = [len(i) for i in self.comid_network.values()]
        arg_sort_l = np.argsort(lengths)[::-1]
        self.comid_network["length"] = list(np.array(lengths)[arg_sort_l])
        c = list(comid_network.keys())
        self.comid_network["comid_start"] = list(np.array(c)[arg_sort_l])
        return

    def map_complete_network_down_up(self, huc_number: int = 4):
        """
        DESCRIPTION:
        ------------
            Map comid network from the headwaters to the terminal nodes.
        ________________________________________________________________________

        Args:
        ------------
            :param huc_number: int, Default 4
                HUC number to be extracted.
        """
        # Extract headwaters
        data_term = self.data_info[self.data_info["terminalfl"] == 1]
        terminal_paths = np.unique(self.data_info["terminalpa"])
        # data_term = data_term.sort_values(by='totdasqkm', ascending=False)
        # start_comids = np.array(data_term.index)
        # terminal_paths = data_term['terminalpa'].values

        # Fill values of comid
        comid_table = copy.deepcopy(self.data_info)
        c_comid = self.data_info.index.values
        self.linking_network.loc[c_comid, "huc12"] = comid_table.loc[
            c_comid, "reachcode"
        ]
        self.linking_network.loc[c_comid, "huc10"] = comid_table.loc[
            c_comid, "huc10"
        ]
        self.linking_network.loc[c_comid, "huc08"] = comid_table.loc[
            c_comid, "huc08"
        ]
        self.linking_network.loc[c_comid, "huc06"] = comid_table.loc[
            c_comid, "huc06"
        ]
        self.linking_network.loc[c_comid, "huc04"] = comid_table.loc[
            c_comid, "huc04"
        ]
        self.linking_network.loc[c_comid, "huc02"] = comid_table.loc[
            c_comid, "huc02"
        ]
        self.linking_network.loc[c_comid, "xm_m"] = comid_table.loc[
            c_comid, "xm_m"
        ]
        self.linking_network.loc[c_comid, "ym_m"] = comid_table.loc[
            c_comid, "ym_m"
        ]
        self.linking_network.loc[c_comid, "xup_m"] = comid_table.loc[
            c_comid, "xup_m"
        ]
        self.linking_network.loc[c_comid, "yup_m"] = comid_table.loc[
            c_comid, "yup_m"
        ]
        self.linking_network.loc[c_comid, "xdown_m"] = comid_table.loc[
            c_comid, "xdown_m"
        ]
        self.linking_network.loc[c_comid, "ydown_m"] = comid_table.loc[
            c_comid, "ydown_m"
        ]

        huc_n_value = np.unique(self.linking_network[f"huc{huc_number:02d}"])
        self.comid_network["huc_list"] = huc_n_value
        self.comid_network.update({str(huc_n): [] for huc_n in huc_n_value})

        time1 = time.time()
        for huc_n in huc_n_value:
            if huc_number == 12:
                key_val = "reachcode"
            else:
                key_val = f"huc{huc_number:02d}"
            # Extract only values of the huc_n
            subset = self.data_info[self.data_info[key_val] == huc_n]
            terminal_paths = np.unique(subset["terminalpa"])
            # pbar = tqdm(
            #     total=len(terminal_paths),
            #     desc=f"Ext. huc {huc_n}",
            # )
            for i_tp, tp in enumerate(terminal_paths):
                # Extract comids that include the terminal path
                comid_table = subset[subset["terminalpa"] == tp]
                # Sort by drinage area
                comid_table = comid_table.sort_values(
                    by="totdasqkm", ascending=False
                )
                # Remove where streamorde and streamcalc are different
                comid_table = comid_table[
                    comid_table["streamorde"] == comid_table["streamcalc"]
                ]
                # Extract comids that have not been extracted
                linking_network = self.linking_network.loc[comid_table.index, :]
                comid_table = comid_table[
                    linking_network["extracted_comid"] == 0
                ]

                # Get starting comid
                st = comid_table.index
                if len(st) == 0:
                    continue
                st = st[0]
                # self.logger.info(f"Extracting comid {st}")
                comid_table = subset[
                    subset["streamorde"] == subset["streamcalc"]
                ]
                comid_network = self._recursive_upstream_exploration(
                    st, comid_table, huc_number=huc_number
                )
                # pbar.update(i_tp)

            # pbar.close()

            lengths = [len(i) for i in comid_network.values()]
            total_length = np.sum(lengths)
            # pbar = tqdm(total=len(subset), desc=" Ext. Extra Nodes")
            while total_length < len(subset):
                linking_network = self.linking_network.loc[subset.index, :]
                linking_network = linking_network[
                    self.linking_network["extracted_comid"] == 0
                ]
                if len(linking_network) == 0:
                    break
                comid_table = subset.loc[linking_network.index, :]
                # sort by drainage area
                comid_table = comid_table.sort_values(
                    by="totdasqkm", ascending=False
                )
                # Remove where streamorde and streamcalc are different
                comid_table = comid_table[
                    comid_table["streamorde"] == comid_table["streamcalc"]
                ]
                # Get starting comid
                st = comid_table.index
                if len(st) == 0:
                    break
                st = st[0]
                # self.logger.info(f"Extracting comid {st}")
                comid_network = self._recursive_upstream_exploration(
                    st, comid_table, huc_number=huc_number
                )
                lengths = [len(i) for i in comid_network.values()]
                total_length = np.sum(lengths)
                # pbar.update(total_length)

            # pbar.close()

            # # Clean duplicated comids
            # for key, value in comid_network.items():
            #     indices = np.unique(value, return_index=True)[1]
            #     comid_network[key] = list(np.array(value)[indices])

            # Convert network from terminal to start
            lengths = [len(i) for i in comid_network.values()]
            arg_sort_l = np.argsort(lengths)[::-1]
            comid_network_2 = {
                str(i[-1]): list(i[::-1]) for i in comid_network.values()
            }
            self.comid_network[huc_n] = comid_network_2
            c = list(comid_network_2.keys())
            self.comid_network[huc_n]["comid_start"] = list(
                np.array(c).astype(str)[arg_sort_l]
            )
            self.comid_network[huc_n]["length"] = list(
                np.array(lengths)[arg_sort_l]
            )
        return

    def _recursive_upstream_exploration(
        self, start_comid, comid_table, comid_network={}, huc_number=4
    ):
        """
        DESCRIPTION:
        ------------
            Recursive exploration of the upstream comids.
        ________________________________________________________________________

        Args:
        ------------
            :param start_comid: int
                Starting comid to be extracted.
            :param comid_table: pd.DataFrame,
                Dataframe with the comid table.
            :param comid_network: dict, Default None
                directory to save the comid network (for recursive).
            :param huc_number: int, Default 4
                HUC number to be extracted.
        """
        c_comid = start_comid
        # if huc_number == 12:
        #     ini_huc = comid_table.loc[c_comid, 'reachcode']
        # else:
        #     ini_huc = comid_table.loc[c_comid, f'huc{huc_number:02d}']

        i = 1
        while i < len(comid_table):
            # Update lists
            self.linking_network.loc[c_comid, "huc_n"] = huc_number
            self.linking_network.loc[c_comid, "extracted_comid"] = 1
            c_comid_pos = c_comid
            # --------------------------
            # Get from nodes
            # --------------------------
            from_i = comid_table.loc[c_comid, "fromnode"]
            # --------------------------
            # Extract new comid
            # --------------------------
            c_comid = comid_table.index[comid_table["tonode"] == from_i].values

            self.linking_network.loc[c_comid, "n_tributaries"] = len(c_comid)

            if len(c_comid) == 0:
                break

            # add linking
            # Search for highest stream order
            so = comid_table.loc[c_comid, "streamorde"].values
            if len(so) > 1:
                if so[0] == so[1]:
                    da = comid_table.loc[c_comid, "totdasqkm"].values
                    arg_max_so = np.argmax(da)
                else:
                    arg_max_so = np.argmax(so)
            else:
                arg_max_so = np.argmax(so)

            # Add start comid to the network
            try:
                comid_network[start_comid]
            except KeyError:
                comid_network[start_comid] = []
            if len(c_comid) == 1:
                comid_network[start_comid].append(c_comid[0])
                if c_comid != c_comid_pos:
                    self.linking_network.loc[c_comid, "linking_comid"] = (
                        c_comid_pos
                    )
                c_comid = c_comid[0]
            # If there are more than one comid, add the one with the highest drainage area
            elif len(c_comid) > 1:
                comid_network[start_comid].append(c_comid[arg_max_so])
                for j, c_comid_j in enumerate(c_comid):
                    if c_comid_j != c_comid_pos:
                        self.linking_network.loc[c_comid_j, "linking_comid"] = (
                            c_comid_pos
                        )
                    if j == arg_max_so:
                        continue
                    comid_network = self._recursive_upstream_exploration(
                        c_comid[j], comid_table, comid_network
                    )
                    # Test recursive to see if the other network exists
                    # try:
                    #     comid_network = self._recursive_upstream_exploration(
                    #         c_comid[j], comid_table, comid_network)
                    # except KeyError:
                    #     continue

                c_comid = c_comid[arg_max_so]

            i += 1

        return comid_network

    def map_complete_reach(
        self,
        start_comid: str,
        huc_number: int = 4,
        do_not_overlap: bool = True,
    ) -> Tuple[Union[list, np.ndarray], list]:
        """
        DESCRIPTION:
        ------------
            Separate complete reach comid from the ToNode and FromNode
            values
        ________________________________________________________________________

        Args:
        ------------
            :param start_comid: str,
                Start comid value.
            :param huc_number: int, Default 4
                Path of the current reach.
            :param do_not_overlap: bool, Default True
                If True, the reach extracted will not overlap with existing
                reaches extracted previously.
            :return reach: list,
                List of comids for the values
            :rtype reach: Union[list, np.ndarray]
            :return huc_n: list,
                List of huc numbers for the values
        """
        comid = np.array(self.data_info.index)
        huc_n = self.data_info.loc[start_comid, f"huc{huc_number:02d}"]

        data_info = self.data_info[
            self.data_info[f"huc{huc_number:02d}"] == huc_n
        ]

        c_comid_prev = copy.deepcopy(start_comid)
        c_comid = start_comid
        comid_network = np.zeros(len(comid))
        i = 0
        i_overlap = 0
        max_overlapping = 1  # Maximum number of overlaps
        while True:
            # self.logger.info(f"{i} {start_comid} Iterating comid {c_comid}")
            self.linking_network.loc[c_comid, "huc_n"] = huc_number
            # --------------------------
            # Get to and from nodes
            # --------------------------
            to_i = data_info.loc[c_comid, "tonode"]
            # --------------------------
            # Save Data
            # --------------------------
            comid_network[i] = c_comid
            self.linking_network.loc[c_comid, "extracted_comid"] = 1

            # --------------------------
            # Extract new comid
            # --------------------------
            c_comid = data_info.index[data_info["fromnode"] == to_i].values
            if isinstance(c_comid, str):
                c_comid = [c_comid]
            if len(c_comid) == 0:
                break
            elif c_comid[0] == c_comid_prev:
                break
            else:
                # --------------------------
                # Iterate over the network
                # --------------------------
                # self.logger.info(f"{i} {start_comid} Next comid {c_comid[0]}")
                c_comid = c_comid[0]
                self.linking_network.loc[c_comid_prev, "linking_comid"] = (
                    c_comid
                )
                c_comid_prev = copy.deepcopy(c_comid)
            # --------------------------
            # Check overlapping
            # --------------------------
            if do_not_overlap:
                # Check three overlapping comids
                linking = self.linking_network.loc[c_comid, "linking_comid"]
                if linking != 0:
                    # Sum in the overlapping
                    i_overlap += 1
                if i_overlap == max_overlapping + 1:
                    i_overlap = 0
                    break
            i += 1
        comid_network = comid_network[comid_network != 0]
        comid_network = comid_network.astype(int)
        comid_network = comid_network.astype(str)
        return comid_network, huc_n

    def map_coordinates(self, comid_list, file_coords):
        """
        DESCRIPTION:
        ------------
            Map Coordinates and additional data to the comid_list
        ________________________________________________________________________

        Args:
        ------------
            :param comid_list: list,
                List with comid values
            :param file_coords: str,
                File name where the coordiantes will be saved.
        """
        timeg = time.time()
        comid_list = np.array(comid_list)
        huc_04s = self.data_info.loc[comid_list, "huc04"].values
        slope = self.data_info.loc[comid_list, "slope"].values
        so_values = self.data_info.loc[comid_list, "streamorde"]
        try:
            da_t = self.data_info.loc[comid_list, "totdasqkm"].values
        except KeyError:
            da_t = np.ones_like(comid_list) * np.nan
        try:
            da_inc = self.data_info.loc[comid_list, "areasqkm"].values
        except KeyError:
            da_inc = np.ones_like(comid_list) * np.nan
        try:
            da_hw = self.data_info.loc[comid_list, "hwnodesqkm"].values
        except KeyError:
            da_hw = np.ones_like(comid_list) * np.nan
        start_comid = self.data_info.loc[comid_list, "startflag"].values
        # lengthkm = self.data_info.loc[comid_list, 'lengthkm'].values
        # cm to m
        max_elev = self.data_info.loc[comid_list, "maxelevsmo"].values / 100
        # Generate Loop
        data = {}
        for huc in self.huc_04:
            # Load coordinates
            time1 = time.time()
            c_all = comid_list[huc_04s == huc]
            indices_c = np.unique(c_all, return_index=True)[1]
            c_all = np.array([c_all[i] for i in sorted(indices_c)])
            if self.pre_loaded_coords:
                coordinates = copy.deepcopy(self.coords_all)
                keys = [i for i in c_all]
                coordinates = {i: np.array(coordinates[i]) for i in keys}
            else:
                # Load File
                if file_coords.split(".")[-1] == "hdf5":
                    keys = [str(i) for i in c_all]
                    coordinates = FM.load_data(f"{file_coords}", keys=keys)
                    coordinates = {i: np.array(coordinates[i]) for i in keys}
                else:
                    coordinates = FM.load_data(f"{file_coords}")
            length_reach = np.array(
                [RF.get_reach_distances(coordinates[i].T)[-1] for i in c_all]
            )
            length_reach = np.hstack([0, length_reach])
            cum_length_reach = np.cumsum(length_reach)
            # print('loading coordinates')
            # utl.toc(time1)
            # -----------------
            # Get coordinates
            # -----------------
            # append coordinates
            time1 = time.time()
            lengths = [len(coordinates[i][0]) for i in c_all]
            xx = [item for i in c_all for item in coordinates[i][0]]
            yy = [item for i in c_all for item in coordinates[i][1]]
            indices = np.unique(xx, return_index=True)[1]
            x = np.array([xx[i] for i in sorted(indices)])
            y = np.array([yy[i] for i in sorted(indices)])
            # print('appending coordinates')
            # utl.toc(time1)
            # ------------------------------------
            # Calculate distance along the river
            # ------------------------------------
            time1 = time.time()
            # Check if the complete reach is lower than 3
            x_coord = np.vstack((x, y)).T
            s = RF.get_reach_distances(x_coord)
            # print('calculating distance along the river')
            # utl.toc(time1)
            # ------------------------------------
            # Calculate Drainage Area
            # ------------------------------------
            time1 = time.time()
            # Calculate accumulated DA
            da_initial = np.zeros(len(da_t) + 1)
            # Set initial DA
            if start_comid[0] == 1:
                if da_hw[0] > 0:
                    da_initial[0] = da_hw[0]
                elif da_inc[0][0] != da_t[0]:
                    da_initial[0] = da_inc[0][0]
                elif da_t[0] > 0:
                    da_initial[0] = 0.1 * da_t[0]
                else:
                    cond = da_t > 0
                    cond = np.where(da_t > 0)[0]
                    if len(cond) > 0:
                        da_t[da_t <= 0] = 0.1 * da_t[cond[0]]
                        da_initial[0] = da_t[0]
                    else:
                        da_t[da_t <= 0] = 1e-5
                        da_initial[0] = da_t[0]
            else:
                if da_inc[0] > 0 and da_inc[0] != da_t[0]:
                    da_initial[0] = da_t[0] - da_inc[0]
                else:
                    da_initial[0] = da_t[0]
            # Calculate rest of the DA
            da_initial[1:] = da_t

            # Interpolate values
            intp = interpolate.interp1d(
                cum_length_reach, da_initial, fill_value="extrapolate"
            )
            da = intp(s)

            # Calculate Width with Wilkerson et al. (2014)
            w = RF.calculate_channel_width(da)
            # print('calculating DA and width')
            # utl.toc(time1)
            # -----------------------------
            # Additional values
            # -----------------------------
            # Add variables
            time1 = time.time()
            comid_values = [i for i in c_all for item in coordinates[i][0]]
            comid_values = np.array([comid_values[i] for i in sorted(indices)])
            so = [so_values[i] for i in c_all for item in coordinates[i][0]]
            so = np.array([so[i] for i in sorted(indices)])
            # print('adding variables')
            # utl.toc(time1)
            # -----------------------------
            # Include Elevation
            # -----------------------------
            time1 = time.time()
            z = np.zeros(x.shape) * np.nan

            # Store data
            time1 = time.time()
            # print('storing data')
            data = {
                "comid": comid_values,
                "x": x,
                "y": y,
                "z": z,
                "s": s,
                "so": so,
                "da_sqkm": da,
                "w_m": w,
            }
            data = pd.DataFrame.from_dict(data)
            data.set_index("comid", inplace=True)
            # print('storing data')
            # utl.toc(time1)

        # print('Time general')
        # utl.toc(timeg)
        return data
