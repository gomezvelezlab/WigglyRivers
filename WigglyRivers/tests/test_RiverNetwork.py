# -*- coding: utf-8 -*-
# _____________________________________________________________________________
# _____________________________________________________________________________
#
#                       Coded by Daniel Gonzalez Duque
#                           Last revised 2024-05-17
# _____________________________________________________________________________
# _____________________________________________________________________________

"""
______________________________________________________________________________

 DESCRIPTION:
   Tests for RiverNetwork.py script
______________________________________________________________________________
"""
# Libraries
from pathlib import Path

from ..rivers.RiversNetwork import RiverDatasets

THIS_FOLDER = Path(__file__).parent.resolve()
TEST_DATA_DIR = THIS_FOLDER / "data_test"


def test_load_river_dataset():
    rivers = RiverDatasets()
    rivers.load_river_network(
        TEST_DATA_DIR / "test_data.hdf5",
        fn_meanders_database=TEST_DATA_DIR / "meander_database.csv",
        fn_tree_scales=TEST_DATA_DIR / "tree_scales.p",
        fn_tree_scales_database=TEST_DATA_DIR / "tree_scales_database.feather",
    )

    river_id = "Idealized River Transect ($\lambda=[50,100,200,500]$)"

    river = rivers[river_id]
