# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 22/02/2022
# _____________________________________________________________________________
# _____________________________________________________________________________

# ------------------------
# Importing Modules
# ------------------------ 
# Manipulate Data
import time
# System

# System
# ------------------
# Other Modules
# ------------------
from .functions import function_temp as ft


# ------------------
# Classes
# ------------------
class Method:
    """
    Class Description

    ===================== =====================================================
    Attribute             Description
    ===================== =====================================================
    Attribute_1            Attribute description
    ===================== =====================================================

    The following are the methods of the class.

    ===================== =====================================================
    Methods               Description
    ===================== =====================================================
    Method_1              Method description
    ===================== =====================================================
    """
    def __init__(self, x):
        """
        """
        # -----------
        # Attribute
        # ----------
        # Private Attribute
        self._x = x
        # Run function
        return

    def method(self):
        return self._x


class Celsius:
    def __init__(self, temperature=0):
        self.temperature = temperature

    def to_fahrenheit(self):
        temperature = ft.to_fahrenheit(self.temperature)
        return temperature

    @property
    def temperature(self):
        print("Getting value...")
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        print("Setting value...")
        if value < -273.15:
            raise ValueError("Temperature below -273 is not possible")
        self._temperature = value