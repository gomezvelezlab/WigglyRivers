# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez-Duque
#                           Last revised 2021-01-15
# _____________________________________________________________________________
# _____________________________________________________________________________
"""

The classExceptions are included in this script.
"""
import warnings
import functools


# File Exceptions
class FileNotFoundError(Exception):
    pass


class FormatError(Exception):
    pass


class VariableNotFoundError(Exception):
    pass


class HUCNotFoundError(Exception):
    pass


class SmallMeanderError(Exception):
    pass


def deprecated(replacement=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if replacement == None:
                warnings.warn(
                    f"Function '{func.__name__}' is deprecated and "
                    f"will be removed in a future version.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if replacement:
                warnings.warn(
                    f"Function '{func.__name__}' is deprecated and "
                    f"will be removed in a future version. Please use "
                    f"'{replacement}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            return func(*args, **kwargs)

        return wrapper

    return decorator
