# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#                               Last revised 2022-09-03
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
______________________________________________________________________________

 DESCRIPTION:
   This class extracts the complete reaches obtained with the model
______________________________________________________________________________
"""
# -----------
# Libraries
# -----------
# System Management
import copy
import logging
import time
from typing import Iterable
from typing import Tuple
from typing import Union
# Data Management
import numpy as np
from scipy import optimize
from scipy import integrate
from scipy import signal
from scipy import interpolate
from scipy.interpolate import UnivariateSpline
import pandas as pd
# MeanderCONUS Packages
from pyMeander.utilities import classExceptions as CE
from pyMeander.utilities import filesManagement as FM
from pyMeander.meander import MeanderFunctions as MF
from pyMeander.utilities import utilities as utl
from . import RiverFunctions as RF


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
    path_out              path to save the files.
    save_format           save format of the information, the formats are:\n
                          'p': pickle.\n
                          'json': json type file.\n
                          'mat': MATLAB type file.\n
                          'csv': csv type file.\n
                          'txt': txt type file.\n
    nhd_tables            NHD tables to be loaded, by default it will load the
                          'NHDPlusFlowlineVAA', 'NHDPlusEROMMA',
                          'NHDPlusIncrPrecipMA', 'NHDPlusIncrTempMA'
    ===================== =====================================================

    The following are the methods of the class.

    ===================== =====================================================
    Methods               Description
    ===================== =====================================================
    read_mt_file          read in a MT file [ EDI | XML | j ]
    ===================== =====================================================


    Examples
    -------------------
    :Read from NHD GBD: ::

        >>> import
    """
    def __init__(self, data, comid_id='nhdplusid', logger=None, **kwargs):
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
            self._logging.info(f'Starting logger {self.__class__.__name__}')
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
        self.huc_04 = np.unique(self.data_info['huc04'].values)
        self._save_format = 'csv'
        self.__save_formats = FM.get_save_formats()

        # Change parameters
        valid_kwargs = ['save_format']
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            k = f'_{k}'
            setattr(self, k, v)

        # --------------------
        # Data Management
        # --------------------
        try:
            self.linking_network = data.loc[
                :, ['nhdplusid', 'startflag', 'within_waterbody']]
        except:
            self.logger.warning('No within_waterbody column found')
            self.linking_network = data.loc[
                :, ['nhdplusid', 'startflag']]
        self.linking_network.set_index('nhdplusid', inplace=True)
        self.linking_network['extracted_comid'] = [0]*len(self.linking_network)
        self.linking_network['linking_comid'] = [0]*len(self.linking_network)
        self.linking_network['huc_n'] = [0]*len(self.linking_network)
        data_hw = self.data_info[self.data_info['startflag'] == 1]
        start_comids = data_hw.index.values
        self.comid_network = {str(st): [] for st in start_comids}
        self.extracted_comids = []

        # ------------
        # NHD Data
        # ------------
        self.coords_all = None
        return

    # --------------------------
    # get functions
    # --------------------------
    # @property
    # def path_coordinates(self):
    #     """ path for data saving"""
    #     return self._path_coordinates

    @property
    def save_format(self):
        """save format of the files"""
        return self._save_format

    @property
    def logger(self):
        """logger for debbuging"""
        return self._logging
    # --------------------------
    # set functions
    # --------------------------
    @save_format.setter
    def save_format(self, save_format):
        """set save format of the files"""
        if save_format not in self.__save_formats:
            self.logger.error(f"format '{save_format}' not implemented. "
                              f"Use any of these formats "
                              f"{self.__save_formats}")
            raise CE.FormatError(f"format '{save_format}' not implemented. "
                                 f"Use formats 'p', 'mat', 'json', 'txt', or"
                                 f"'csv'")
        else:
            self.logger.info(f"Setting save_format to '{save_format}'")
            self._save_format = save_format

    # --------------------------
    # Core functions
    # --------------------------
    def load_coords(self, path_coords: str) -> None:
        """
        DESCRIPTION:
            Load coordinates from a file.
        _______________________________________________________________________
        INPUT:
            :param path_coords: str
                Path to the coordinates file.
        """
        self.logger.info(f'Loading coordinates from {path_coords}')
        self.coords_all = FM.load_data(path_coords)
        self.pre_loaded_coords = True
        return

    def map_complete_network(self,
                             start_comids: Union[list, np.ndarray, None]=None,
                             huc_number: int=4,
                             max_num_comids: Union[int, None]=None,
                             cut_comid_number: int=3,
                             do_not_overlap: bool=True,
                             start_over: bool=True) -> None:
        """
        DESCRIPTION:
            Map the entire database.
        _______________________________________________________________________
        INPUT:
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
        _______________________________________________________________________
        OUTPUT:
            :return reach: list,
                List of comids for the values
        """
        if start_comids is None:
            # Extract headwaters
            data_hw = self.data_info[self.data_info['startflag'] == 1]
            # data_hw = data_hw.sort_values(by='totdasqkm')
            data_hw = data_hw.sort_values(by='dnhydroseq', ascending=False)
            start_comids = np.array(data_hw.index)
        if isinstance(start_comids, str) or isinstance(start_comids, float):
            start_comids = [start_comids]
        
        if max_num_comids is None:
            max_num_comids = len(start_comids)
        
        # Look for the ones that have been not extracted
        linking_start = self.linking_network[
            self.linking_network['startflag'] == 1]
        
        start_comids = start_comids[linking_start['extracted_comid'] == 0]
        # Loop over the comids
        self.extracted_comids = [0]*max_num_comids
        self.time = [0]*max_num_comids
        i_data = 0
        i_extracted = 0
        while i_data <= len(start_comids):
            time_1 = time.time()
            if i_extracted >= max_num_comids:
                break
            start_comid = start_comids[i_data]
            self.logger.info(f"Extracting comid {start_comid}")
            comid_network, _ = self.map_complete_reach(
                start_comid, huc_number, do_not_overlap=do_not_overlap)
            # Check if the comid_network is less than the cut_comid_number
            if len(comid_network) <= cut_comid_number:
                i_data += 1
                self.linking_network.loc[start_comid, 'extracted_comid'] = -1
                continue
            self.comid_network[str(comid_network[0])] = comid_network
            self.extracted_comids[i_extracted] = start_comid
            self.linking_network.loc[start_comid, 'extracted_comid'] = 1
            self.time[i_extracted] = (time.time() - time_1)/60
            self.logger.info(f"Time elapsed: {self.time[i_extracted]} min")
            i_data += 1
            i_extracted += 1
        

        lengths = [len(i) for i in self.comid_network.values()]
        arg_sort_l = np.argsort(lengths)[::-1]
        self.comid_network['length'] = list(np.array(lengths)[arg_sort_l])
        c = list(comid_network.keys())
        self.comid_network['comid_start'] = list(np.array(c)[arg_sort_l])
        return
    
    def map_complete_network_down_up(self, huc_number: int=4):
        """
        DESCRIPTION:
            Map the entire database. It will do exploration from the terminal
            nodes to the headwaters. Saving the information in each reach.
        _______________________________________________________________________
            INPUT:
            
        """
        # Extract headwaters
        data_term = self.data_info[self.data_info['terminalfl'] == 1]
        terminal_paths = np.unique(self.data_info['terminalpa'])
        # data_term = data_term.sort_values(by='totdasqkm', ascending=False)
        # start_comids = np.array(data_term.index)
        # terminal_paths = data_term['terminalpa'].values

        time1 = time.time()
        for tp in terminal_paths:

            # Extract comids that include the terminal path
            comid_table = self.data_info[
                self.data_info['terminalpa'] == tp]
            # Sort by drinage area
            comid_table = comid_table.sort_values(
                by='totdasqkm', ascending=False)
            # Remove where streamorde and streamcalc are different
            comid_table = comid_table[
                comid_table['streamorde'] == comid_table['streamcalc']]
            # Extract comids that have not been extracted
            linking_network = self.linking_network.loc[comid_table.index, :]
            comid_table = comid_table[linking_network['extracted_comid'] == 0]

            # Get starting comid
            st = comid_table.index
            if len(st) == 0:
                continue
            st = st[0]
            # self.logger.info(f"Extracting comid {st}")
            comid_table = self.data_info[
                self.data_info['streamorde'] == self.data_info['streamcalc']]
            comid_network = self._recursive_upstream_exploration(
                st, comid_table, huc_number=huc_number)
        
        lengths = [len(i) for i in comid_network.values()]
        total_length = np.sum(lengths)
        while total_length < len(self.data_info):
            linking_network = self.linking_network[
                self.linking_network['extracted_comid'] == 0]
            if len(linking_network) == 0:
                break
            comid_table = self.data_info.loc[linking_network.index, :]
            # sort by drainage area
            comid_table = comid_table.sort_values(
                by='totdasqkm', ascending=False)
            # Remove where streamorde and streamcalc are different
            comid_table = comid_table[
                comid_table['streamorde'] == comid_table['streamcalc']]
            # Get starting comid
            st = comid_table.index
            if len(st) == 0:
                break
            st = st[0]
            # self.logger.info(f"Extracting comid {st}")
            comid_network = self._recursive_upstream_exploration(
                st, comid_table, huc_number=huc_number)
            lengths = [len(i) for i in comid_network.values()]
            total_length = np.sum(lengths)

            
        # Convert network from terminal to start
        lengths = [len(i) for i in comid_network.values()]
        arg_sort_l = np.argsort(lengths)[::-1]
        comid_network_2 = {str(i[-1]): list(i[::-1])
                         for i in comid_network.values()}
        self.comid_network = comid_network_2
        c = list(comid_network_2.keys())
        self.comid_network['comid_start'] = list(
            np.array(c).astype(float)[arg_sort_l])
        self.comid_network['length'] = list(np.array(lengths)[arg_sort_l])

        # ---------------------------------------------------------------------
        # Test

        # i = 0
        # i_c = 0
        # comids_keys = self.comid_network['comid_start']
        # c_e = comids_keys[i_c]
        # comid_table = self.data_info[
        #     self.data_info['terminalpa'] == terminal_paths[i]]
        # so = self.data_info.loc[c_e, 'streamorde']
        # print(so)

        # x_up = self.data_info.loc[:, 'xup_deg'].values
        # y_up = self.data_info.loc[:, 'yup_deg'].values
        # x_down = self.data_info.loc[:, 'xdown_deg'].values
        # y_down = self.data_info.loc[:, 'ydown_deg'].values

        # x_up_net = self.data_info.loc[comid_network_2[str(c_e)], 'xup_deg'].values
        # y_up_net = self.data_info.loc[comid_network_2[str(c_e)], 'yup_deg'].values
        # x_down_net = self.data_info.loc[comid_network_2[str(c_e)], 'xdown_deg'].values
        # y_down_net = self.data_info.loc[comid_network_2[str(c_e)], 'ydown_deg'].values

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 10))
        # plt.plot([x_up, x_down], [y_up, y_down], '-b')
        # plt.plot([x_up_net, x_down_net], [y_up_net, y_down_net], '-r')
        # plt.gca().set_aspect('equal', adjustable='box')
        # plt.show()

        # for i_c in range(1, len(arg_sort_l)):
        #     i_c = 4
        #     c_e = comids_keys[i_c]
        #     x_up_net = self.data_info.loc[
        #         comid_network_2[str(c_e)], 'xup_deg'].values
        #     y_up_net = self.data_info.loc[
        #         comid_network_2[str(c_e)], 'yup_deg'].values
        #     x_down_net = self.data_info.loc[
        #         comid_network_2[str(c_e)], 'xdown_deg'].values
        #     y_down_net = self.data_info.loc[
        #         comid_network_2[str(c_e)], 'ydown_deg'].values

        #     plt.plot([x_up_net, x_down_net], [y_up_net, y_down_net], '-r')
        # ---------------------------------------------------------------------

        return

    def _recursive_upstream_exploration(self, start_comid, comid_table,
                                        comid_network={}, huc_number=4):

        comid_network[start_comid] = [start_comid]
        c_comid_pos = start_comid
        c_comid = start_comid

        i = 1
        while i < len(comid_table):
            # Update lists
            self.linking_network.loc[c_comid, 'huc_n'] = huc_number
            self.linking_network.loc[c_comid, 'extracted_comid'] = 1
            c_comid_pos = c_comid
            # --------------------------
            # Get from nodes
            # --------------------------
            from_i = comid_table.loc[c_comid, 'fromnode']
            # --------------------------
            # Extract new comid
            # --------------------------
            c_comid = comid_table.index[
                comid_table['tonode'] == from_i].values
            if len(c_comid) == 0:
                break
            
            # add linking
            # Search for highest stream order
            so = comid_table.loc[c_comid, 'streamorde'].values
            if len(so) > 1:
                if so[0] == so[1]:
                    da = comid_table.loc[c_comid, 'totdasqkm'].values
                    arg_max_so = np.argmax(da)
                else:
                    arg_max_so = np.argmax(so)
            else:
                arg_max_so = np.argmax(so)
            if len(c_comid) == 1:
                comid_network[start_comid].append(c_comid[0])
                if c_comid != c_comid_pos:
                    self.linking_network.loc[
                        c_comid, 'linking_comid'] = c_comid_pos
                c_comid = c_comid[0]
            elif len(c_comid) > 1:
                comid_network[start_comid].append(c_comid[arg_max_so])
                for j, c_comid_j in enumerate(c_comid):
                    if c_comid_j != c_comid_pos:
                        self.linking_network.loc[c_comid_j,
                                                'linking_comid'] = c_comid_pos
                    if j == arg_max_so:
                        continue
                    comid_network = self._recursive_upstream_exploration(
                        c_comid[j], comid_table, comid_network)
                
                c_comid = c_comid[arg_max_so]

            i += 1

        return comid_network

    def map_complete_reach(
            self, start_comid: str,
            huc_number: int=4,
            do_not_overlap: bool=True) -> Tuple[Union[list, np.ndarray], list]:
        """
        DESCRIPTION:
            Separate complete reach comid from the ToNode and FromNode
            values
        _______________________________________________________________________
        INPUT:
            :param start_comid: str,
                Start comid value.
            :param path: int, Default 4
                Path of the current reach.
            :param do_not_overlap: bool, Default True
                If True, the reach extracted will not overlap with existing
                reaches extracted previously.
        _______________________________________________________________________
        OUTPUT:
            :return reach: list,
                List of comids for the values
            :rtype reach: Union[list, np.ndarray]
            :return huc_n: list,
                List of huc numbers for the values
        """
        comid = np.array(self.data_info.index)
        huc_n = self.data_info.loc[start_comid, f'huc{huc_number:02d}']

        data_info = self.data_info[
            self.data_info[f'huc{huc_number:02d}'] == huc_n]

        c_comid_prev = start_comid
        c_comid = start_comid
        comid_network = np.zeros(len(comid))
        i = 0
        i_overlap = 0
        max_overlapping = 1  # Maximum number of overlaps
        while True:
            # self.logger.info(f"{i} {start_comid} Iterating comid {c_comid}")
            self.linking_network.loc[c_comid, 'huc_n'] = huc_number
            # --------------------------
            # Get to and from nodes
            # --------------------------
            to_i = data_info.loc[c_comid, 'tonode']
            # --------------------------
            # Save Data
            # --------------------------
            comid_network[i] = c_comid
            self.linking_network.loc[
                c_comid, 'extracted_comid'] = 1
                
            # --------------------------
            # Extract new comid
            # --------------------------
            c_comid = data_info.index[
                data_info['fromnode'] == to_i].values
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
                self.linking_network.loc[
                    c_comid_prev, 'linking_comid'] = c_comid
                c_comid_prev = copy.deepcopy(c_comid)
            # --------------------------
            # Check overlapping
            # --------------------------
            if do_not_overlap:
                # Check three overlapping comids
                linking = self.linking_network.loc[c_comid, 'linking_comid']
                if linking != 0:
                    # Sum in the overlapping
                    i_overlap += 1
                if i_overlap == max_overlapping:
                    i_overlap = 0
                    break
            i += 1
        comid_network = comid_network[comid_network != 0]
        return comid_network, huc_n

    def map_coordinates(self, comid_list, file_coords):
        """
        Map Coordinates and additional data to the comid_list
        """
        timeg = time.time()
        comid_list = np.array(comid_list).astype(float)
        huc_04s = self.data_info.loc[comid_list, 'huc04'].values
        slope = self.data_info.loc[comid_list, 'slope'].values
        so_values = self.data_info.loc[comid_list, 'streamorde']
        try:
            da_t = self.data_info.loc[comid_list, 'totdasqkm'].values
        except KeyError:
            da_t = np.ones_like(comid_list) * np.nan
        try:
            da_inc = self.data_info.loc[comid_list, 'areasqkm'].values
        except KeyError:
            da_inc = np.ones_like(comid_list) * np.nan
        try:
            da_hw = self.data_info.loc[comid_list, 'hwnodesqkm'].values
        except KeyError:
            da_hw = np.ones_like(comid_list) * np.nan
        start_comid = self.data_info.loc[comid_list, 'startflag'].values
        # lengthkm = self.data_info.loc[comid_list, 'lengthkm'].values
        # cm to m
        max_elev = self.data_info.loc[comid_list, 'maxelevsmo'].values / 100
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
                keys = [str(i) for i in c_all]
                coordinates = {float(i): coordinates[i] for i in keys}
            else:
                # Load File
                if file_coords.split('.')[-1] == 'hdf5':
                    keys = [str(i) for i in c_all]
                    coordinates = FM.load_data(f'{file_coords}', keys=keys)
                    coordinates = {float(i): coordinates[i] for i in keys}
                else:
                    coordinates = FM.load_data(f'{file_coords}')
            length_reach = np.array([
                RF.get_reach_distances(coordinates[i].T)[-1]for i in c_all])
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
                elif da_inc[0] != da_t[0]:
                    da_initial[0] = da_inc[0]
                elif da_t[0] > 0:
                    da_initial[0] = 0.1*da_t[0]
                else:
                    cond = da_t > 0
                    cond = np.where(da_t > 0)[0]
                    if len(cond) > 0:
                        da_t[da_t <= 0] = 0.1*da_t[cond[0]]
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
            intp = interpolate.interp1d(cum_length_reach, da_initial,
                                        fill_value='extrapolate')
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
            comid_values = [
                float(i) for i in c_all for item in coordinates[i][0]]
            comid_values = np.array([comid_values[i] for i in sorted(indices)])
            so = [so_values[float(i)] for i in c_all for item in
                  coordinates[i][0]]
            so = np.array([so[i] for i in sorted(indices)])
            # print('adding variables')
            # utl.toc(time1)
            # -----------------------------
            # Include Elevation
            # -----------------------------
            time1 = time.time()
            z = np.zeros(x.shape)
            i_cc = 0
            for i_c, c in enumerate(c_all):
                if i_c == 0:
                    z_max = max_elev[i_c]
                else:
                    z_max = z[i_cc]

                z[i_cc:i_cc + lengths[i_c]] = (
                        z_max - (s[i_cc:i_cc + lengths[i_c]] - s[i_cc])
                        * slope[i_c])
                i_cc += lengths[i_c] - 1
            
            # print('including elevation')
            # utl.toc(time1)

            # Store data
            time1 = time.time()
            # print('storing data')
            data = {'comid': comid_values, 'x': x, 'y': y,
                    'z': z, 's': s,
                    'so': so, 'da_sqkm': da, 'w_m': w}
            data = pd.DataFrame.from_dict(data)
            data.set_index('comid', inplace=True)
            # print('storing data')
            # utl.toc(time1)
        
        # print('Time general')
        # utl.toc(timeg)
        return data

    @staticmethod
    def fit_splines(data, method='min'):
        """
        Fit splines to the coordinates
        """
        # Extract data
        comid = np.array(data.index)
        so = data['so'].values
        s = data['s'].values
        x = data['x'].values
        y = data['y'].values
        z = data['z'].values
        da = data['da_sqkm'].values
        w = data['w_m'].values
        # ---------------
        # Get poly S
        # ---------------
        if method == 'min':
            diff_s = np.min(np.diff(s))
        elif method == 'geometric_mean':
            diff_s = 10**np.mean(np.log10(np.diff(s)))
        else:
            raise ValueError(f"method '{method} not implemented."
                             f"Please use 'min' or 'geometric_mean'")
        s_poly = np.arange(s[0], s[-1] + diff_s, diff_s)
        # ------------------
        # Generate Splines
        # -----------------
        x_spl = UnivariateSpline(s, x, k=3, s=0, ext=0)
        y_spl = UnivariateSpline(s, y, k=3, s=0, ext=0)
        z_spl = UnivariateSpline(s, z, k=1, s=0, ext=0)
        f_comid = interpolate.interp1d(s, comid,
                                       fill_value=(comid[0], comid[-1]),
                                       kind='previous', bounds_error=False)
        f_so = interpolate.interp1d(s, so,
                                       fill_value=(so[0], so[-1]),
                                       kind='previous', bounds_error=False)
        f_da = interpolate.interp1d(s, da,
                                    fill_value=(da[0], da[-1]),
                                    kind='previous', bounds_error=False)
        f_w = interpolate.interp1d(s, w,
                                   fill_value=(w[0], w[-1]),
                                   kind='previous', bounds_error=False)
        # ------------------
        # Create points
        # -----------------
        x_poly = x_spl(s_poly)
        y_poly = y_spl(s_poly)
        z_poly = z_spl(s_poly)
        comid_poly = f_comid(s_poly)
        so_poly = f_so(s_poly)
        da_poly = f_da(s_poly)
        w_poly = f_w(s_poly)
        # ------------------
        # Create data
        # -----------------
        data_fitted = {
            's_poly': s_poly ,'x_poly': x_poly, 'y_poly': y_poly,
            'z_poly': z_poly, 'comid_poly': comid_poly,
            'so_poly': so_poly, 'da_sqkm_poly': da_poly, 'w_m_poly': w_poly}

        return data_fitted

    def smooth_data(self, data, poly_order=2,
                    savgol_window=1, gaussian_window=1):

        # --------------------------
        # Extract data
        # --------------------------
        try:
            x = data['x_poly']
            y = data['y_poly']
            s = data['s_poly']
        except KeyError:
            raise KeyError('Data does not have the polynomial fit.'
                           'Run fit_spline(data) first before smoothing')

        # --------------------------
        # Smooth Data
        # --------------------------
        x_smooth, y_smooth, s_smooth = RF.smooth_data(
            x, y, s, poly_order=poly_order, savgol_window=savgol_window,
            gaussian_window=gaussian_window)

        # -----------------
        # Save Data
        # -----------------
        data_smooth = {'x_smooth': x_smooth, 'y_smooth': y_smooth,
                       's_smooth': s_smooth}
        return data_smooth



