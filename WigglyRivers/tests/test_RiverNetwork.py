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
import numpy as np

from ..rivers.RiversNetwork import RiverDatasets
from ..rivers import RiverFunctions as RF

THIS_FOLDER = Path(__file__).parent.resolve()
TEST_DATA_DIR = THIS_FOLDER / "data_test"


def create_data():
    def explore_kinoshita_values(
        theta_0, lambda_value, theta_s, theta_f, n, plot_flag=False
    ):
        x_k, y_k, data = RF.kinoshita_curve_zolezzi(
            theta_0=theta_0,
            lambda_value=lambda_value,
            theta_s=theta_s,
            theta_f=theta_f,
            n=n,
        )
        w_m = np.ones_like(x_k)
        # Create River Object
        rivers_k = RiverDatasets()
        rivers_k.add_river(
            "0",
            x_k,
            y_k,
            w_m=w_m,
            resample_flag=True,
            kwargs_resample={},
            scale_by_width=False,
        )
        results = {
            "rivers": rivers_k,
        }
        return results

    theta_0 = 110
    theta_s = 0.344
    theta_f = 0.031
    lambda_value = [50, 100, 200, 500]
    n = 5
    variables = ["rivers"]

    results_lambda = {i: [] for i in variables}

    for i in range(len(lambda_value)):
        result = explore_kinoshita_values(
            theta_0 * np.pi / 180, lambda_value[i], theta_s, theta_f, n
        )

        if i == 0:
            results_lambda["rivers"] = result["rivers"]
        else:
            results_lambda["rivers"]["0"].x = np.concatenate(
                (
                    results_lambda["rivers"]["0"].x[:-2],
                    result["rivers"]["0"].x
                    + results_lambda["rivers"]["0"].x[-1],
                )
            )
            results_lambda["rivers"]["0"].y = np.concatenate(
                (results_lambda["rivers"]["0"].y[:-2], result["rivers"]["0"].y)
            )
            results_lambda["rivers"]["0"].w_m = np.concatenate(
                (results_lambda["rivers"]["0"].w_m, result["rivers"]["0"].w_m)
            )
            results_lambda["rivers"]["0"].s = np.concatenate(
                (
                    results_lambda["rivers"]["0"].s,
                    result["rivers"]["0"].s
                    + np.max(results_lambda["rivers"]["0"].s),
                )
            )

    x = results_lambda["rivers"]["0"].x
    y = results_lambda["rivers"]["0"].y
    w_m = np.ones_like(x)

    rivers_lambda = RiverDatasets()
    river_id = r"Idealized River Transect ($\lambda=[50,100,200,500]$)"
    rivers_lambda.add_river(
        river_id,
        x,
        y,
        w_m=w_m,
        resample_flag=True,
        kwargs_resample={},
        scale_by_width=False,
    )

    return rivers_lambda, river_id


def test_load_river_dataset():
    rivers = RiverDatasets()
    rivers.load_river_network(
        TEST_DATA_DIR / "test_data.hdf5",
        fn_meanders_database=TEST_DATA_DIR / "meander_database.csv",
        fn_tree_scales=TEST_DATA_DIR / "tree_scales.p",
        fn_tree_scales_database=TEST_DATA_DIR / "tree_scales_database.feather",
    )

    river_id = "Idealized River Transect ($\lambda=[50,100,200,500]$)"

    assert rivers[river_id]


def test_curvature_derivatives():

    theta_0 = 115 * np.pi / 180  # radians
    n = 35
    m_points = 3500
    x_k, y_k, data = RF.kinoshita_curve_zolezzi(
        theta_0=theta_0,
        lambda_value=200,
        theta_s=0.0,
        theta_f=0,
        n=n,
        m_points=m_points,
    )
    c_original = data["c"]
    # ==============================
    # WigglyRivers
    # ==============================
    id_river = "Kinoshita Idealized"
    rivers_k = RiverDatasets()
    #   The problem is on the creation of the tree. Why?
    rivers_k.add_river(
        id_river,
        x_k,
        y_k,
        resample_flag=True,
        # kwargs_resample={'smooth': 1e-2},
        kwargs_resample={},
        scale_by_width=False,
    )
    rivers_k[id_river].calculate_curvature()
    c_wiggly = rivers_k[id_river].c

    np.testing.assert_allclose(c_original, c_wiggly[1:], atol=1e-2)


def test_curvature_grad():

    theta_0 = 115 * np.pi / 180  # radians
    n = 35
    m_points = 3500
    x_k, y_k, data = RF.kinoshita_curve_zolezzi(
        theta_0=theta_0,
        lambda_value=200,
        theta_s=0.0,
        theta_f=0,
        n=n,
        m_points=m_points,
    )
    c_original = data["c"]
    # ==============================
    # WigglyRivers
    # ==============================
    id_river = "Kinoshita Idealized"
    rivers_k = RiverDatasets()
    #   The problem is on the creation of the tree. Why?
    rivers_k.add_river(
        id_river,
        x_k,
        y_k,
        resample_flag=False,
        # kwargs_resample={'smooth': 1e-2},
        kwargs_resample={},
        scale_by_width=False,
    )
    rivers_k[id_river].calculate_curvature()
    c_wiggly = rivers_k[id_river].c

    np.testing.assert_allclose(c_original, c_wiggly[1:], atol=1e-2)


def test_cwt_calculation():

    # Load test data
    rivers = RiverDatasets()
    rivers.load_river_network(
        TEST_DATA_DIR / "test_data.hdf5",
        fn_meanders_database=TEST_DATA_DIR / "meander_database.csv",
        fn_tree_scales=TEST_DATA_DIR / "tree_scales.p",
        fn_tree_scales_database=TEST_DATA_DIR / "tree_scales_database.feather",
    )

    # Create dataset
    rivers_test, river_id = create_data()

    rivers_test[river_id].calculate_curvature()

    # --------------------
    # Calculate CWT
    # --------------------
    rivers_test[river_id].extract_cwt_tree()

    cwt_data = rivers.rivers[river_id].cwt_wave_c
    cwt_test = rivers_test.rivers[river_id].cwt_wave_c

    np.testing.assert_allclose(cwt_data, cwt_test, atol=1e-2)


def test_meander_identification():

    # Load test data
    rivers = RiverDatasets()
    rivers.load_river_network(
        TEST_DATA_DIR / "test_data.hdf5",
        fn_meanders_database=TEST_DATA_DIR / "meander_database.csv",
        fn_tree_scales=TEST_DATA_DIR / "tree_scales.p",
        fn_tree_scales_database=TEST_DATA_DIR / "tree_scales_database.feather",
    )

    # Create dataset
    rivers_test, river_id = create_data()

    rivers_test[river_id].calculate_curvature()

    # --------------------
    # Calculate CWT
    # --------------------
    rivers_test[river_id].extract_cwt_tree()
    # -----------------------------
    # Prune by peak power
    # -----------------------------
    rivers_test[river_id].prune_tree_by_peak_power()
    # -----------------------------
    # Prune by sinuosity
    # -----------------------------
    rivers_test[river_id].prune_tree_by_sinuosity(1.01)
    # -----------------------------
    # Add meander to database
    # -----------------------------
    rivers_test[river_id].add_meanders_from_tree_scales(
        overwrite=True, clip="downstream", bounds_array_str="extended"
    )

    database_original = rivers.rivers[river_id].database
    database_test = rivers_test.rivers[river_id].database

    # Check if the length of the databases is +- 2

    assert len(database_original) - len(database_test) in [0, 2]
