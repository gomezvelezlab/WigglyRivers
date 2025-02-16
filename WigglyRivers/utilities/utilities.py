# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 2021-01-14
# _____________________________________________________________________________
# _____________________________________________________________________________
"""

The functions given on this package allow the user to manipulate and create
functions from the computer.


"""
# ------------------------
# Importing Modules
# ------------------------
# System
import os
import time
import pathlib as pl
from typing import Union


def unzip_file(zip_file: Union[pl.Path, str], path_output: Union[pl.Path, str]):
    """unzip a file to a specific path.

    Args:
        zip_file (Union[pl.Path, str]): Path to the zip file.
        path_output (Union[pl.Path, str]): Path to extract the zip file.
    """
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(path_output)


def cr_folder(path: Union[pl.Path, str]):
    """Create a folder in a specific path.

    Args:
        path (Union[pl.Path, str]): Path to create the folder.
    """

    if path != "":
        # Verify if the path already exists
        if not os.path.exists(path):
            os.makedirs(path)


def get_folders(path: Union[pl.Path, str]) -> list:
    """get folders in a specific path.

    Args:
        path (Union[pl.Path, str]): get folders in a specific path.

    Returns:
        list: list of folders in the path.
    """
    return next(os.walk(path))[1]


def toc(time1: float):
    """print the time of execution.

    Args:
        time1 (float): time when execution started given by time.time().
    """
    dif = time.time() - time1
    if dif >= 3600 * 24:
        print(f"====\t{dif/3600/24:.4f} days\t ====")
    elif dif >= 3600:
        print(f"====\t{dif/3600:.4f} hours\t ====")
    elif dif >= 60:
        print(f"====\t{dif/60:.4f} minutes\t ====")
    else:
        print(f"====\t{dif:.4f} seconds\t ====")


def fix_widget_error():
    """
    Fix FigureWidget - 'mapbox._derived' Value Error.
    Adopted from:

    https://github.com/plotly/plotly.py/issues/2570#issuecomment-738735816
    """
    import shutil
    import pkg_resources

    pkg_dir = os.path.dirname(
        pkg_resources.resource_filename("plotly", "plotly.py")
    )

    basedatatypesPath = os.path.join(pkg_dir, "basedatatypes.py")

    backup_file = basedatatypesPath.replace(".py", "_bk.py")
    # find if backup file exists
    if os.path.exists(backup_file):
        # copy the backup file as current file
        shutil.copyfile(backup_file, basedatatypesPath)
    else:
        # Save a backup copy of the original file
        shutil.copyfile(basedatatypesPath, backup_file)

    # read basedatatypes.py
    with open(basedatatypesPath, "r") as f:
        lines = f.read()

    find = (
        "if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):"
    )

    replace = """if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):
                if key_path_str == "mapbox._derived":
                    return"""

    # add new text
    lines = lines.replace(find, replace)

    # overwrite old 'basedatatypes.py'
    with open(basedatatypesPath, "w") as f:
        f.write(lines)


# fix_widget_error()
