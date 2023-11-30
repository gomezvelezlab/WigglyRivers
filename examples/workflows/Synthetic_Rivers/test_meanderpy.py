
# Importing Packages
import os
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display, HTML
from IPython.display import display
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean
from scipy import interpolate
import statsmodels.api as sm

import meanderpy as mp

# Importing WigglyRivers Packages
from WigglyRivers import RiverDatasets, RiverTransect
from WigglyRivers import RiverFunctions as RF
from WigglyRivers import ExtractNHDPlusHRData as NHD
from WigglyRivers.utilities import utilities as utl
from WigglyRivers.utilities import filesManagement as FM
from WigglyRivers.utilities import graphs
from WigglyRivers import Logger
from WigglyRivers import WaveletTreeFunctions as WTFunc
from WigglyRivers import iwavelet

# Set seed for numpy
np.random.seed(0)
data = FM.load_data(
    'examples/workflows/kinoshita/synthetic_rivers/synthetic_rivers_w_noise.hdf5')
river_ids = list(data.keys())
# ------------------------------
# Arguments for the noise data
# ------------------------------
kwargs_resample = {
    '0':{'smooth': 1e2},
    '1':{'smooth': 1e2},
    '2':{'smooth': 1e2},
    '3':{'smooth': 1e2},
    '0_noise':{'smooth': 1e2},
    '1_noise':{'smooth': 1e3},
    '2_noise':{'smooth': 1e3},
    '3_noise':{'smooth': 1e3},
    }

# ------------------------------
# Start Logger
# ------------------------------
logger = Logger(console=True)
# ------------------------------
# Create Rivers class
# ------------------------------
rivers = RiverDatasets(logger=logger)
# ------------------------------
# Add Rives into class
# ------------------------------
i = 0
for river_id in river_ids:
    # ------------------------------
    # Add original river 
    # ------------------------------
    x_ch = data[river_id]['x']
    y_ch = data[river_id]['y']
    w_m = data[river_id]['w_m']
    rivers.add_river(river_id, x_ch, y_ch, w_m=w_m, resample_flag=True,
                     kwargs_resample=kwargs_resample[river_id],
                     scale_by_width=True)