import os
import matplotlib.pyplot as plt   # plots
import numpy as np
import scipy.sparse as sp
import h5py
import warnings
import json
import pickle

from liblibra_core import *
import util.libutil as comn
from libra_py import units, data_conv
import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot

from libra_py.workflows.nbra import rpi

istep = 12000000
fstep = 28000000 + 1

path_to_save_Hvibs = '../01-sampling_dynamics/res/'
nstates = 2

model_params = {"timestep":0, "icond":0,  "model0":0, "nstates":nstates}

E = []
for step in range(istep, fstep):
    energy_filename = F"{path_to_save_Hvibs}/Hvib_{step}_re.npz"
    energy_mat = sp.load_npz(energy_filename)
    E.append( np.array( np.diag( energy_mat.todense() ) ) )
E = np.array(E)

St = []
for step in range(istep, fstep):
    St_filename = F"{path_to_save_Hvibs}/St_{step}_re.npz"
    St_mat = sp.load_npz(St_filename)
    St.append( np.array( St_mat.todense() ) )
St = np.array(St)

NAC = []
Hvib = []
for c, step in enumerate(range(istep, fstep)):
    nac_filename = F"{path_to_save_Hvibs}/Hvib_{step}_im.npz"
    nac_mat = sp.load_npz(nac_filename)
    NAC.append( np.array( nac_mat.todense() ) )
    Hvib.append( np.diag(E[c, :])*(1.0+1j*0.0)  - (0.0+1j)*nac_mat[:, :] )

NAC = np.array(NAC)
Hvib = np.array(Hvib)

model_params.update({"E": E, "St": St, "NAC": NAC, "Hvib": Hvib})

# Setting the coherent Ehrenfest propagation. Define the argument-dependent part first.
dyn_general = {"nstates":nstates, "which_adi_states":range(nstates), "which_dia_states":range(nstates)}

dyn_general.update({"ntraj":1, "mem_output_level":2, "progress_frequency":1,  
                    "properties_to_save":["timestep", "time", "se_pop_adi"], "prefix":"./", "isNBRA": 0,
                    "ham_update_method":2, "ham_transform_method":0, "time_overlap_method":0, "nac_update_method":0,
                    "hvib_update_method":0, "force_method":0, "rep_force":1, "hop_acceptance_algo":0, "momenta_rescaling_algo":0, 
                    "rep_tdse":1, "electronic_integrator":2, "tsh_method":-1, "decoherence_algo":-1, "decoherence_times_type":-1, 
                    "do_ssy":0, "dephasing_informed":0})

# Explicitly set some defaults (or reasonable value!), to suppress warnings
dyn_general.update({"rep_sh":1, "rep_lz":0, "rep_ham":0, "enforce_state_following":0, "direct_init": 0,
                    "enforced_state_indxed":0, "do_phase_correction":0, "phase_correction_tol":1e-10,
                    "max_number_attempts":10,  "total_energy":0.01, "tcnbra_nu_therm":0.01,
                    "thermally_corrected_nbra":0, "tcnbra_nhc_size":1, "tcnbra_do_nac_scaling":1, 
                    "use_Jasper_Truhlar_criterion":1, "fssh2_revision": 0, 
                    "fssh3_size_option":0, "fssh3_approach_option":0, "fssh3_decomp_option":0, 
                    "fssh3_dt": 0.001, "fssh3_max_steps": 1000, "fssh3_err_tol": 1e-7,
                    "decoherence_C_param":0.1, "decoherence_eps_param":1.0, 
                    "MK_alpha": 0.0, "MK_verbosity":0, "sdm_norm_tolerance":1e-4,
                    "use_xf_force":0, "project_out_aux":0, "use_td_width": 0, "coherence_threshold": 0.01,
                    "entanglement_opt": 0, "ETHD3_alpha":0.01, "ETHD3_beta":0.01, "Temperature":300.0
                   })

# Electronic DOF
elec_params = {"ndia":nstates, "nadi":nstates, "verbosity":-1, "init_dm_type":0, "init_type":1, "rep":1,"istate":0}

# Nuclear DOF - these parameters don't matter much in the NBRA calculations
nucl_params = {"ndof":1, "init_type":3, "q":[-10.0], "p":[0.0], "mass":[2000.0], "force_constant":[0.01], "verbosity":-1,
               "q_width": [1.0], "p_width": [1.0]}

dyn_params = dict(dyn_general)
with open('dyn_params.pkl', 'wb') as f:
    pickle.dump(dyn_params, f)

with open('model_params.pkl', 'wb') as f:
    pickle.dump(model_params, f)

with open('elec_params.pkl', 'wb') as f:
    pickle.dump(elec_params, f)

with open('nucl_params.pkl', 'wb') as f:
    pickle.dump(nucl_params, f)

Nt = 8000000 # ~ 20 ps

## npatches used in this work are 1000, 6250, 12500, 20000
rpi_params = {'run_slurm': True, 'submit_template': 'submit_template.slm', 'submission_exe': 'sbatch', 
              'run_python_file': 'run_template.py', 'python_exe': 'python',
              'nsteps': Nt, 'iconds': [0, 2000000, 4000000, 6000000, 8000000], 'npatches': 1000, 'nstates': 2, 'dt': 0.1*units.fs2au, 
              'prefix': 'out_'  
             }

rpi.generic_patch_dyn(rpi_params, dyn_params, None, model_params, elec_params, nucl_params)
