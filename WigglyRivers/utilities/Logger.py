# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by: co, lossycomp,
#                                 Jesus Gomez-Velez and
#                                 Daniel Gonzalez-Duque
#
#                               Last revised 2022-01-17
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
import os
import logging

# ------------------
# Logging
# ------------------
# Set logger
logging.basicConfig(handlers=[logging.NullHandler()])


# ------------------
# Class
# ------------------
class Logger:
    """
    This class is to perform the logging of the package for debugging
    or information.

    ===================== =====================================================
    Attribute             Description
    ===================== =====================================================
    console               Boolean, show logger information in the terminal
                          prompt
    file                  Export a file with the log information.
    level                 Level of logger, the levels are
    format                Format to present the logging results.
    ===================== =====================================================

    The following are the methods of the class.

    ===================== =====================================================
    Methods               Description
    ===================== =====================================================
    set_logger            Set the logger
    ===================== =====================================================


    Examples
    -------------------
    :Set Logger: ::

    >>> logger = Logger(console=True)
    """

    def __init__(
            self, console=False, file=None, level='DEBUG',
            format='%(asctime)s[%(levelname)s] %(funcName)s: %(message)s'):
        """
        Class constructor
        """
        # ------------------------
        # Create logger
        # ------------------------
        self._logging = logging.getLogger(self.__class__.__name__)
        self._logging.setLevel(logging.INFO)

        # ------------------------
        # Select level
        # ------------------------
        if level.upper() == 'DEBUG':
            level = logging.DEBUG
        elif level.upper() == 'CRITICAL':
            level = logging.CRITICAL
        elif level.upper() == 'ERROR':
            level = logging.ERROR
        elif level.upper() == 'WARNING':
            level = logging.WARNING
        elif level.upper() == 'INFO':
            level = logging.INFO
        elif level.upper() == 'NOTSET':
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
            self, console=False, file=None, level=logging.DEBUG,
            format='%(asctime)s[%(levelname)s] %(funcName)s: %(message)s'):
        """set logger handlers"""
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
        self._logging.info(f'Starting log')
        if file is not None:
            self._logging.info(f'Log will be saved in {file}')

    def close_logger(self):
        """Close current logger"""
        self.logger.info('Close Logger')
        handlers = self.logger.handlers[:]
        for handler in handlers:
            self.logger.removeHandler(handler)
            handler.close()
            return

    # --------------------------
    # Print values
    # --------------------------
    def info(self, msg):
        self.logger.info(msg)
        return

    def warning(self, msg):
        self.logger.warning(msg)
        return

    def error(self, msg):
        self.logger.error(msg)
        return

    def critical(self, msg):
        self.logger.critical(msg)
        return

    def debug(self, msg):
        self.logger.debug(msg)
        return
