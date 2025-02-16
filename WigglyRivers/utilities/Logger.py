# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: Daniel Gonzalez-Duque and
#                                 Jesus Gomez-Velez
#
#
#                               Last revised 2025-02-13
# _____________________________________________________________________________
# _____________________________________________________________________________
"""
______________________________________________________________________________

 DESCRIPTION:
   This class acts as the logger for all the classes
______________________________________________________________________________
"""
# -----------
# Libraries
# -----------
# System Management
from typing import Union
import os
import logging
import pathlib as pl

# ------------------
# Logging
# ------------------
# Set logger
logging.basicConfig(handlers=[logging.NullHandler()])


# ------------------
# Class
# ------------------
class Logger:

    def __init__(
        self,
        console: bool = False,
        file: Union[str, pl.Path, None] = None,
        level: Union[int, str] = "DEBUG",
        format: str = "%(asctime)s[%(levelname)s] %(funcName)s: %(message)s",
    ):
        """Logger constructor.

        Args:
            console (bool, optional): show logger information in the terminal.
                Defaults to False.
            file (Union[str, pl.Path, None], optional): Export logger
                information to file. If None no file will be created.
                Defaults to None.
            level (Union[int, str], optional): level of logger, the levels are:
                DEBUG, CRITICAL, ERROR, WARNING, INFO, NOTSET. Defaults to "DEBUG".
            format (str, optional): format for logger message.
                Defaults to "%(asctime)s[%(levelname)s] %(funcName)s: %(message)s".
        """
        # ------------------------
        # Create logger
        # ------------------------
        self._logging = logging.getLogger(self.__class__.__name__)
        self._logging.setLevel(logging.INFO)

        # ------------------------
        # Select level
        # ------------------------
        if level.upper() == "DEBUG":
            level = logging.DEBUG
        elif level.upper() == "CRITICAL":
            level = logging.CRITICAL
        elif level.upper() == "ERROR":
            level = logging.ERROR
        elif level.upper() == "WARNING":
            level = logging.WARNING
        elif level.upper() == "INFO":
            level = logging.INFO
        elif level.upper() == "NOTSET":
            level = logging.NOTSET
        # ------------------------
        # Set logger
        # ------------------------
        self.set_logger(console, file, level, format)

    # --------------------------
    # get functions
    # --------------------------
    @property
    def logger(self):
        """logger for debbuging"""
        return self._logging

    # --------------------------
    # set functions
    # --------------------------
    def set_logger(
        self,
        console: bool = False,
        file: Union[str, pl.Path, None] = None,
        level: Union[int, str] = logging.DEBUG,
        format: str = "%(asctime)s[%(levelname)s] %(funcName)s: %(message)s",
    ):
        """Setting logging

        Args:
            console (bool, optional): show logger information in the terminal.
                Defaults to False.
            file (Union[str, pl.Path, None], optional): Export logger
                information to file. If None no file will be created.
                Defaults to None.
            level (Union[int, str], optional): level of logger, the levels are:
                DEBUG, CRITICAL, ERROR, WARNING, INFO, NOTSET. Defaults to "DEBUG".
            format (str, optional): format for logger message.
                Defaults to "%(asctime)s[%(levelname)s] %(funcName)s: %(message)s".
        """
        # ------------------------
        # Select level
        # ------------------------
        if isinstance(level, str):
            if level.upper() == "DEBUG":
                level = logging.DEBUG
            elif level.upper() == "CRITICAL":
                level = logging.CRITICAL
            elif level.upper() == "ERROR":
                level = logging.ERROR
            elif level.upper() == "WARNING":
                level = logging.WARNING
            elif level.upper() == "INFO":
                level = logging.INFO
            elif level.upper() == "NOTSET":
                level = logging.NOTSET
        formatter = logging.Formatter(format)
        if console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            self._logging.addHandler(console_handler)
        if file is not None:
            # Remove previous file
            if os.path.isfile(file):
                os.remove(file)
            file_handler = logging.FileHandler(file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self._logging.addHandler(file_handler)
        self._logging.info("Starting log")
        if file is not None:
            self._logging.info(f"Log will be saved in {file}")

    def close_logger(self):
        """Close current logger"""
        self.logger.info("Close Logger")
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
            return

    # --------------------------
    # Print values
    # --------------------------
    def info(self, msg: str):
        """Logger message

        Args:
            msg (str): message
        """
        self.logger.info(msg)
        return

    def warning(self, msg: str):
        """Logger message

        Args:
            msg (str): message
        """
        self.logger.warning(msg)
        return

    def error(self, msg: str):
        """Logger message

        Args:
            msg (str): message
        """
        self.logger.error(msg)
        return

    def critical(self, msg: str):
        """Logger message

        Args:
            msg (str): message
        """
        self.logger.critical(msg)
        return

    def debug(self, msg: str):
        """Logger message

        Args:
            msg (str): message
        """
        self.logger.debug(msg)
        return
