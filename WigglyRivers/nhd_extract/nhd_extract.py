# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque
#                               Last revised 2021-09-14
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
   This class open the data and do preprocessing of the data.
"""
# -----------
# Libraries
# -----------
# System Management
import logging
import os

# Data Management
import numpy as np
import pandas as pd

# Modules of the package
from ..utilities import classExceptions as CE
from ..utilities import filesManagement as FM
from ..utilities import utilities as utl

# ------------------
# Logging
# ------------------
# Set logger
logging.basicConfig(handlers=[logging.NullHandler()])


# ------------------
# Class
# ------------------
class ExtractNHDPlusHRData:
    """
    Basic class to load and pre-processes NHD data. This class opens the NHDPlus
    High-Resolution data and extracts the information needed to run the
    meander characterization code.

    The NHDPlus High-Reolution data can be downloaded from the following link:
    https://www.usgs.gov/national-hydrography/nhdplus-high-resolution

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
    set_logger            Set the logger to show every process.
    get_data_from_nhd_gdb Extract coordinates and tables from NHD GDB.
    load_nhd_data         Loads the tables and coordinates from the NHD GDB.
    ===================== =====================================================
    """

    def __init__(
        self,
        path_output,
        name="file",
        comid_name="NHDPlusID",
        logger=None,
        **kwargs,
    ):
        """
        Class constructor
        """
        # -------------------
        # Logger
        # -------------------
        if logger is None:
            self._logging = logging.getLogger(self.__class__.__name__)
        else:
            self._logging = logger
            self._logging.info(f"Start Logger in {self.__class__.__name__}")
        # ------------------------
        # Attribute management
        # ------------------------
        # Default data
        # Path management
        self._path_output = path_output
        self._save_format = "feather"
        self.name = name
        self.comid_name = comid_name

        # Tables for NHD
        self._nhd_tables = [
            "NHDPlusFlowlineVAA",
            "NHDPlusEROMMA",
        ]
        # self._nhd_tables += [
        #     f'NHDPlusIncrPrecipMM{i:02d}' for i in range(1, 13)
        #     ]
        # self._nhd_tables += [
        #     f'NHDPlusIncrTempMM{i:02d}' for i in range(1, 13)
        # ]

        # Change parameters
        valid_kwargs = ["save_format", "nhd_tables"]
        for k, v in zip(kwargs.keys(), kwargs.values()):
            if k not in valid_kwargs:
                raise TypeError("Invalid keyword argument %s" % k)
            k = f"_{k}"
            setattr(self, k, v)

        # ------------------
        # Create Folder
        # ------------------
        utl.cr_folder(path_output)
        # ------------------
        # Other Attributes
        # ------------------
        self.__save_formats = FM.get_save_formats()

        # ------------
        # NHD Data
        # ------------
        self.table = None
        self.coords_all = None

    # --------------------------
    # get functions
    # --------------------------
    @property
    def path_output(self):
        """path for data saving"""
        return self._path_output

    @property
    def save_format(self):
        """save format of the files"""
        return self._save_format

    @property
    def nhd_tables(self):
        """nhd tables to be extracted"""
        return self._nhd_tables

    @property
    def logger(self):
        """logger for debbuging"""
        return self._logging

    # --------------------------
    # setters
    # --------------------------
    @path_output.setter
    def path_output(self, path_output):
        """set path for data saving"""
        self.logger.info(f"Setting path_output to '{path_output}'")
        self._path_output = path_output

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

    @nhd_tables.setter
    def nhd_tables(self, nhd_tables):
        """set nhd tables to be extracted"""
        self.logger.info(f"Setting nhd_tables to '{nhd_tables}'")
        if not isinstance(nhd_tables, list):
            nhd_tables = [nhd_tables]
        self._nhd_tables = nhd_tables

    # --------------------------
    # read and save files
    # --------------------------
    @staticmethod
    def _get_xy(shapefile):
        try:
            c = shapefile.coords.xy
        except NotImplementedError:
            try:
                c = shapefile.xy
            except NotImplementedError:
                c = []
        return c

    def _get_coordinates(self, shapefile, comid):
        """
        Description:
        --------------
            Get starting and ending coordinates from shapefile Line.
        ________________________________________________________________________

        Args:
        -----
            :param shapefile: geopandas.geometry,
                Geometry of the shapefile.
            :type shapefile: geopandas.geometry
            :param comid: str
                comid of the reach.
            :type comid: str
            :return: (coords, coords_all)
                coords: start, middle, and end coordinates
                coords_all: all coordinates.
            :rtype: tuple(list, dict)
        """
        try:
            c = shapefile.coords.xy
        except NotImplementedError:
            try:
                c = shapefile.xy
            except NotImplementedError:
                if shapefile.geom_type == "MultiLineString":
                    geoms = shapefile.geoms
                    c_values = [self._get_xy(geom) for geom in geoms]
                    for i_c, c in enumerate(c_values):
                        if i_c == 0:
                            x = c[0]
                            y = c[1]
                        else:
                            x = np.append(x, c[0])
                            y = np.append(y, c[1])

                    min_x_y = [np.min(x), np.min(y)]
                    max_x_y = [np.max(x), np.max(y)]
                    mid_x_y = [
                        (min_x_y[0] + max_x_y[0]) / 2,
                        (min_x_y[1] + max_x_y[1]) / 2,
                    ]
                    x_new = np.array([min_x_y[0], mid_x_y[0], max_x_y[0]])
                    y_new = np.array([min_x_y[1], mid_x_y[1], max_x_y[1]])
                    c = [x_new, y_new]
                else:
                    self.logger.error(f"comid {comid} has no coordinates")
                    coords = np.zeros(6) * np.nan
                    coords_all = {
                        comid: [np.array([np.nan]), np.array([np.nan])]
                    }
                    return coords, coords_all
        coords = np.array(
            [
                c[0][0],
                c[-1][0],
                c[0][int((len(c[0]) / 2))],
                c[-1][int((len(c[-1]) / 2))],
                c[0][-1],
                c[-1][-1],
            ]
        )
        x_all = np.array(c[0])
        y_all = np.array(c[-1])
        coords_all = {comid: [x_all, y_all]}
        return coords, coords_all

    @staticmethod
    def _organize_coordinates(
        coords_dict, comid="NHDPlusID", neglect=None, comid_list=None
    ):
        """

        Organize the coordinates from dict to lists including
        the reference comid.

        Parameters
        ----------
        :param coords_dict: dict,
            dictionary with all the coordiantes.
        :type coords_dict: dict
        :param comid: str,
            comid in the dictionary name.
        :type comid: str
        :param neglect: list,
            List with comids that will not be included.
        :type neglect: list

        Returns
        -------

        """
        if neglect is None:
            neglect = np.array([])
        # comid_hr = coords_dict[comid]
        if comid_list is None:
            comid_hr = list(coords_dict)
        else:
            comid_hr = comid_list
        labels = ["FType", comid, "projection", "NHDPlusID", "comid", "COMID"]
        length_hr = np.array(
            [coords_dict[c][0].shape[0] for c in comid_hr if c not in labels]
        )
        coords_hr_all = np.zeros((2, np.sum(length_hr))) * np.nan
        comid_hr_all = np.zeros(np.sum(length_hr)) * np.nan
        i = 0
        for i_c, c in enumerate(comid_hr):
            if c in labels:
                continue
            if len(neglect[neglect == float(c)]) > 0:
                continue
            coords_hr_c = np.array(coords_dict[c])
            comid_hr_c = np.ones(coords_hr_c[0].shape) * float(c)
            coords_hr_all[:, i : i + len(comid_hr_c)] = coords_hr_c
            comid_hr_all[i : i + len(comid_hr_c)] = comid_hr_c
            i += len(comid_hr_c)
        if len(neglect) > 0:
            coords_hr_all = coords_hr_all[:, ~np.isnan(comid_hr_all)]
            comid_hr_all = comid_hr_all[~np.isnan(comid_hr_all)]
        return coords_hr_all, comid_hr_all

    def get_data_from_nhd_gbd(
        self,
        file_data,
        flowlines="NHDFlowline",
        comid="NHDPlusID",
        tables=None,
        projection=None,
    ):
        """

        Description:
        ------------
            Get table data from the NHD GBD.
        ________________________________________________________________________

        Args:
        ------
            :param file_data: str,
                GBD data file.
            :type file_data: str
            :param flowlines: str,
                Layer in the NHD data set that has the Flowlines.
            :type flowlines: str
            :param tables: str,
                Additional tables that will be asociated to the flowlines.
            :type tables: str
            :param comid: str, default NHDPlusID.
                comid of the NHD files.
            :type comid: str
            :param projection: str, default epsg:2856.
                Projection of the NHD file into geographic coordinates
                to calculate the distances.
            :type projection: str
            :param save_shp: bool, default True.
                Save shapefile with initial table.
            :type save_shp: bool
        """
        # -------------------------------
        # Get tables
        # -------------------------------
        self.logger.info(f"Start: Extracting data from '{file_data}'")
        if tables is None:
            tables = self.nhd_tables

        self.comid_name = comid

        # -------------------------------
        # Open Data
        # -------------------------------
        ext = file_data.split(".")[-1]
        file_data = file_data.replace("\\", "/")
        if ext == "zip":
            self.logger.warning(
                " WARNING: GDB in zip file, loading from zip" " file"
            )
            gbd_file_name = file_data.split("/")[-1].split(".")[-2]
            # get whole path of file_data
            path_data = os.path.abspath(file_data)
            path_data = path_data[0].upper() + path_data[1:]
            path_data = path_data.replace("\\", "/")
            print(path_data)
            # file_data = f'zip://{file_data}!{gbd_file_name}.gdb'
            file_data = f"zip://{path_data}!{gbd_file_name}.gdb"
            self.logger.info(f" Openning {file_data}")

        # Load the flowlines
        self.logger.info(" Start: Load Shapefile")
        shapefile = FM.read_gdb(file_data, layer=flowlines)
        shapefile[comid] = shapefile[comid].astype("int64")
        shapefile[comid] = shapefile[comid].astype(str)
        self.logger.info(" Done: Load Shapefile")
        if projection is not None:
            self.logger.info(f" Setting shapefile projection to {projection}")
            try:
                shapefile_projected = shapefile.to_crs(projection)
            except:
                shapefile_projected = shapefile.to_crs({"init": projection})

            coords_all_projected = {
                comid: shapefile_projected[comid].values,
                "FType": shapefile_projected["FType"].values,
                "projection": projection,
            }
            coords_projected = np.zeros((shapefile.shape[0], 6)) * np.nan

        # -------------------------------
        # Get Coordinates
        # -------------------------------
        # Get the start and ending points
        coords_all = {
            comid: shapefile[comid].values,
            "FType": shapefile["FType"].values,
        }
        self.logger.info(" Start: Extraction of coordinates")
        coords = np.zeros((shapefile.shape[0], 6)) * np.nan
        for i in range(shapefile.shape[0]):
            try:
                bounds = shapefile.geometry[i].geoms[0]
            except TypeError:
                bounds = shapefile.geometry[i][0]
            coords[i], coords_all_comid = self._get_coordinates(
                bounds, shapefile[comid].values[i]
            )
            coords_all.update(coords_all_comid)
            if projection is not None:
                # Extract coordinates in projected coordinates
                try:
                    bounds_proj = shapefile_projected.geometry[i].geoms[0]
                except TypeError:
                    bounds_proj = shapefile_projected.geometry[i][0]
                coords_projected[i], coords_all_comid_projected = (
                    self._get_coordinates(
                        bounds_proj, shapefile_projected[comid].values[i]
                    )
                )
                coords_all_projected.update(coords_all_comid_projected)
        self.logger.info(" Done: Extraction of coordinates")

        # -------------------------------
        # Check NHDWaterbody
        # -------------------------------
        # Extract NHDWaterbody
        shapefile_wb = FM.read_gdb(file_data, layer="NHDWaterbody")
        shapefile_wb[comid] = shapefile_wb[comid].astype("int64")
        shapefile_wb[comid] = shapefile_wb[comid].astype(str)
        join_left_intersects_df = shapefile.sjoin(
            shapefile_wb, how="left", predicate="within"
        )
        # Get records that are in the waterbody
        flowlines_within = join_left_intersects_df[
            join_left_intersects_df.index_right.notnull()
        ]
        comid_within = flowlines_within[f"{comid}_left"].values
        # Add flag to the flowlines
        shapefile.set_index(comid, inplace=True)
        shapefile["within_waterbody"] = np.zeros(shapefile.shape[0])
        shapefile.loc[comid_within, "within_waterbody"] = 1
        # -------------------------------
        # Convert to pandas dataframe
        # -------------------------------
        table = pd.DataFrame(shapefile, copy=True)
        # Remove geometry
        table = table.drop("geometry", axis=1)
        # Make COMID the index
        # table = table.set_index(comid)

        # print(coords.shape)
        # Merge coords
        coords_dict = {
            comid: np.array(table.index),
            "xup_deg": coords[:, 0],
            "yup_deg": coords[:, 1],
            "xm_deg": coords[:, 2],
            "ym_deg": coords[:, 3],
            "xdown_deg": coords[:, 4],
            "ydown_deg": coords[:, 5],
        }
        if projection is not None:
            coords_dict.update(
                {
                    "xup_m": coords_projected[:, 0],
                    "yup_m": coords_projected[:, 1],
                    "xm_m": coords_projected[:, 2],
                    "ym_m": coords_projected[:, 3],
                    "xdown_m": coords_projected[:, 4],
                    "ydown_m": coords_projected[:, 5],
                }
            )
            xup_m = coords_dict["xup_m"]
            xdown_m = coords_dict["xdown_m"]
            yup_m = coords_dict["yup_m"]
            ydown_m = coords_dict["ydown_m"]
            length = np.zeros(len(table.index))
            for i_comid, id_comid in enumerate(table.index):
                x = coords_all_projected[id_comid][0]
                y = coords_all_projected[id_comid][1]
                length[i_comid] = np.sum(
                    np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
                )
            sn = length / np.sqrt(
                (xup_m - xdown_m) ** 2 + (yup_m - ydown_m) ** 2
            )
            coords_dict["sinuosity"] = sn
            coords_dict["length_m"] = length

        coords = pd.DataFrame.from_dict(coords_dict)
        coords = coords.set_index(comid)
        table = pd.merge(table, coords, left_index=True, right_index=True)

        # -------------------------------
        # Merge other tables
        # -------------------------------
        self.logger.info(" Start: Merging Tables")
        self.logger.debug(f"Table length: {len(table)}")
        for i_t, t in enumerate(tables):
            # Load new table
            self.logger.info(f"  Merging {t}")
            try:
                t_n = FM.read_gdb(file_data, layer=t)
                t_n[comid] = t_n[comid].astype("int64")
                t_n[comid] = t_n[comid].astype(str)
            except KeyError:
                self.logger.warning(f"  WARNING: Table {t} not in the GDB")
                continue
            t_n = pd.DataFrame(t_n.drop(columns="geometry"), copy=True)
            self.logger.debug(f"Table merge length: {len(t_n)}")
            # Include NaNs if table does not exists
            if len(t_n) == 0:
                self.logger.debug("Filling empty table")
                # get COMIDS
                comid_t_n = np.array(table.index)
                # Extract columns
                columns = list(table.columns)
                columns_t_n = []
                for col in list(t_n.columns):
                    try:
                        columns.index(col)
                        continue
                    except ValueError:
                        columns_t_n.append(col)
                # Extract data
                data_t_n = {
                    i: np.zeros(comid_t_n.shape) * np.nan for i in columns_t_n
                }
                data_t_n[comid] = comid_t_n
                t_n = pd.DataFrame.from_dict(data_t_n)
            t_n = t_n.set_index(comid)
            cols_to_use = t_n.columns.difference(table.columns)
            try:
                table = pd.merge(
                    table,
                    t_n[cols_to_use],
                    left_index=True,
                    right_index=True,
                    how="left",
                )
            except ValueError:
                self.logger.error(f"  ERROR: Could not merge {t}")
            self.logger.debug(f"Table length: {len(table)}")

        self.logger.info(" Done: Merging Tables")
        # -------------------------------
        # Save Tables
        # -------------------------------
        # Save raw tables
        path_output = f"{self.path_output}/tables/"
        table_save = table.reset_index()
        # save_data = {column: table[column].values for column in table.columns}
        name = self.name
        name_out = f"{name}_tables_raw.{self._save_format}"
        self.logger.info(f" Saving {path_output}/{name_out}")
        FM.save_data(table_save, path_output, name_out)

        # Save coordinates
        save_data_coord = coords_all
        save_data_coord.pop("NHDPlusID", None)
        save_data_coord.pop("FType", None)
        save_data_coord.pop("projection", None)
        path_output = f"{self.path_output}/coordinates/"
        name_out = f"{name}_coords_raw.hdf5"
        if self.save_format != "hdf5":
            self.logger.warning(
                "  WARNING: The coordinates are always saved" " as 'hdf5'"
            )
        self.logger.info(f" Saving {path_output}/{name_out}")
        FM.save_data(save_data_coord, path_output, name_out)

        if projection is not None:
            # Save coordinates projected
            save_data_coord = coords_all_projected
            name_out = f'{name}_coords_p_{projection.split(":")[-1]}_raw.hdf5'
            if self.save_format != "hdf5":
                self.logger.warning(
                    "  WARNING: The coordinates are always " "saved as 'hdf5'"
                )
            self.logger.info(f" Saving {path_output}/{name_out}")
            FM.save_data(save_data_coord, path_output, name_out)

        # Load variables
        self.table = table_save
        self.coords_all = save_data_coord
        self.logger.info(f"Done: Extracting data from '{file_data}'")
        return

    def load_nhd_data(self, data_table, data_coords=None, **kwargs):
        """
        DESCRIPTION:
            Get table data from the NHD GBD.
        ________________________________________________________________________
        INPUT:
            :param data_table: str,
                Raw data table.
            :param data_coords: str,
                Coordiantes in the pickle file.
        ________________________________________________________________________
        OUTPUT:
            :return loaded_data: dict,
                dictionary with new data.
        """
        self.logger.info(f"Loading table in {data_table}")
        self.table = FM.load_data(f"{data_table}", **kwargs)
        if data_coords is not None:
            self.logger.info(f"Loading coordiantes in {data_coords}")
            self.coords_all = FM.load_data(f"{data_coords}")
