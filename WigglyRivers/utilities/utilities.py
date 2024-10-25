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


def unzip_file(zip_file, path_output):
    """

    Description:
    ---------------
        Function to Unzip files.

    _______________________________________________________________________

    Args:
    ------
    :param zip_file: str
        file to be unzipped.
    :type zip_file: str
    :param path_output: str
        Path where it will be unzipped
    :type path_output: str
    :return: None
    :rtype: None
    """
    import zipfile

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(path_output)


def cr_folder(path):
    """
    DESCRIPTION:

        This function creates a folder in the given path, if the path does
        not exist then it creates the path itself
    _______________________________________________________________________

    INPUT:
        :param path: A str, Path that needs to be created.
    _______________________________________________________________________
    OUTPUT:
        :return: This function create all the given path.
    """
    if path != "":
        # Verify if the path already exists
        if not os.path.exists(path):
            os.makedirs(path)


def get_folders(path):
    """
    DESCRIPTION:

        This function gets the folders and documents inside a
        specific folder.
    _______________________________________________________________________

    INPUT:
        :param path: A str, Path where the data would be taken.
    _______________________________________________________________________
    OUTPUT:
        :return: A List, List with the folders and files inside
                 the path.
    """
    return next(os.walk(path))[1]


def toc(time1):
    """
    DESCRIPTION:
        Print the time pased over the period selected
    _______________________________________________________________________

    INPUT:
        :param time1: A time.time, time.time() instance.
    _______________________________________________________________________
    OUTPUT:
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

    pkg_dir = os.path.dirname(pkg_resources.resource_filename("plotly", "plotly.py"))

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

    find = "if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):"

    replace = """if not BaseFigure._is_key_path_compatible(key_path_str, self.layout):
                if key_path_str == "mapbox._derived":
                    return"""

    # add new text
    lines = lines.replace(find, replace)

    # overwrite old 'basedatatypes.py'
    with open(basedatatypesPath, "w") as f:
        f.write(lines)


# fix_widget_error()
