import os
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
import h5py
import warnings

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv

from libra_py.workflows.nbra import rpi
import scipy.sparse as sp

## npatches used in this work are 1000, 6250, 12500, 20000
rpi_sum_params = { 'nprocs': 2, 'nsteps': 8000000, 'dt': 0.1*units.fs2au, 'istate': 1, 'nbatches': 5, 'npatches': 1000, 'nstates': 2, 
              'path_to_save_patch': '../03-RPI_PATCH', 'prefix_path':"out_", 'prefix': "POP_NPATCH_" }

rpi.run_sum(rpi_sum_params)
