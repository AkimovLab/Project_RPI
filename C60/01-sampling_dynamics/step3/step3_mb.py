import os
import glob
import numpy as np
import scipy.sparse as sp
from libra_py import units, data_stat, influence_spectrum
import matplotlib.pyplot as plt
from liblibra_core import *
from libra_py.workflows.nbra import step3
import libra_py.packages.cp2k.methods as CP2K_methods

params_mb_sd = {
          'lowest_orbital': 120-30, 'highest_orbital': 120+14, 'num_occ_states': 31, 'num_unocc_states': 14,
          'isUKS': 0, 'number_of_states': 54, 'tolerance': 0.0, 'verbosity': 0, 'use_multiprocessing': True, 'nprocs': 20,
          'is_many_body': True, 'time_step': 1.0, 'es_software': 'cp2k',
          'path_to_npz_files': os.getcwd()+'/../step2/res',
          'logfile_directory': os.getcwd()+'/../step2/all_logfiles',
          'path_to_save_sd_Hvibs': os.getcwd()+'/res-mb-sd-DFT',
          'outdir': os.getcwd()+'/res-mb-sd-DFT', 'start_time': 1200, 'finish_time': 3001, 'sorting_type': 'energy',
         }

step3.run_step3_sd_nacs_libint(params_mb_sd)

