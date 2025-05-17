#!/usr/bin/env python

import sys
import cmath
import math
import os
import h5py
import matplotlib.pyplot as plt   # plots
import numpy as np
import scipy.sparse as sp
import time
import warnings
import argparse

if sys.platform=="cygwin":
    from cyglibra_core import *
elif sys.platform=="linux" or sys.platform=="linux2":
    from liblibra_core import *

import util.libutil as comn
from libra_py import units
from libra_py import units, data_conv #, dynamics_plotting
import libra_py.models.Holstein as Holstein
import libra_py.models.Phenol as Phenol
import libra_py.models.Libra as model_Libra
import libra_py.models.GLVC as GLVC

from libra_py import dynamics_plotting

import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot

import libra_py.dynamics.exact.compute as dvr
import libra_py.dynamics.exact.save as dvr_save

parser = argparse.ArgumentParser()
parser.add_argument('--method_indx', default=0, type=int)
parser.add_argument('--icond_indx', default=0, type=int)
parser.add_argument('--ibatch', default=0, type=int)
args = parser.parse_args()

colors = {}
colors.update({"11": "#CB3310", "12": "#E0603B", "13": "#F08566"}) # red
colors.update({"21": "#0077BB", "22": "#4093D2", "23": "#80B0E9"}) # blue
colors.update({"31": "#009988", "32": "#33B8A8", "33": "#66D7C8"}) # teal
colors.update({"41": "#EE7733", "42": "#F59566", "43": "#FDC399"}) # orange
colors.update({"51": "#00FFFF", "52": "#66FFFF", "53": "#99FFFF"}) # cyan
colors.update({"61": "#EE3377", "62": "#F46695", "63": "#FB99B3"}) # magenta
colors.update({"71": "#AA3377", "72": "#BF6695", "73": "#D399B3"}) # purple
colors.update({"81": "#BBBBBB", "82": "#DDDDDD", "83": "#EEEEEE"}) # grey

clrs_index = [
    "11", "21", "31", "41", "51", "61", "71", "81", 
    "12", "22", "32", "42", "52", "62", "72", "82", 
    "13", "23", "33", "43", "53", "63", "73", "83"
]

class abstr_class:
    pass

def compute_model_nbra(q, params, full_id):
    timestep = params["timestep"]
    nst = params["nstates"]
    obj = abstr_class()
    
    E = params["E"]
    NAC = params["NAC"]
    Hvib = params["Hvib"]
    St = params["St"]

    obj.ham_adi = data_conv.nparray2CMATRIX( np.diag(E[timestep, : ]) )
    obj.nac_adi = data_conv.nparray2CMATRIX( NAC[timestep, :, :] )
    obj.hvib_adi = data_conv.nparray2CMATRIX( Hvib[timestep, :, :] )
    obj.basis_transform = CMATRIX(nst,nst); obj.basis_transform.identity()  #basis_transform
    obj.time_overlap_adi = data_conv.nparray2CMATRIX( St[timestep, :, :] )
    
    return obj

def compute_model(q, params, full_id):

    model = params["model"]
    res = None

    if model==0:
        res = compute_model_nbra(q, params, full_id)
    elif model==1:
        res = GLVC.GLVC(q, params, full_id)
    else:
        pass

    return res

# Spin-boson model
s = units.inv_cm2Ha
freqs = np.array([129.743484,263.643268,406.405874,564.000564,744.784527,
                  961.545250,1235.657107,1606.622430,2157.558683,3100.000000])*s
freqs = freqs.tolist()
g_vals = [0.000563,0.001143,0.001762,0.002446,0.003230,0.004170,0.005358,0.006966,0.009356,0.013444]
eps = 0.02
v0 = 0.001
m = 1836.0 # 1 amu
temperature = 300.0 # K
model_params1 = {"model":1, "model0":1, "nstates":2, "num_osc":10, "coupling_scaling":[1.0, -1.0], 
                 "omega":[freqs]*2, "coupl":[g_vals]*2, "mass":[m]*10, "Ham":[[eps,-v0],[-v0,-eps]],
                 "beta": 1/(units.kB*temperature)}

model_params2 = {"model":2, "model0":2, "nstates":55} # for C60

all_model_params = [model_params1, model_params2]

#################################
# Give the model used an index
model_indx = 1
################################

model_params = all_model_params[model_indx]

NSTATES = model_params["nstates"]
list_states = [x for x in range(NSTATES)]

if model_indx == 0:
    Nt = 8000000 + 1
elif model_indx == 1:
    Nt = 1000 + 1
    
model_params_nbra = dict(model_params)
model_params_nbra.update( {"model":0, "model0":0} )

# Read the data
istep = 1200
fstep = 3000 + 1

path_to_save_Hvibs = "../01-sampling_dynamics/res-mb-sd-DFT"

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

model_params_nbra.update({"E": E, "St": St, "NAC": NAC, "Hvib": Hvib})



# ## Choose a method

# 0: FSSH / 1: mSDM
#################
method_indx = args.method_indx
#################

dyn_nbra = { "nsteps":Nt, "ntraj":250, "nstates":NSTATES,
             "dt":1.0*units.fs2au, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
             "progress_frequency":1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),      
             "mem_output_level":3,
             "properties_to_save":[ "timestep", "time", "Epot_ave", "Ekin_ave", "Etot_ave",
                                   "Cadi", "se_pop_adi", "sh_pop_adi"],
             "prefix":F"fssh_nbra_{model_indx}", "prefix2":F"fssh_nbra_{model_indx}"
           }

#"ham_update_method":2  # recompute only adiabatic Hamiltonian; use with file-based or on-the-fly workflows
#"ham_transform_method":0 # don't do any transforms; usually for NBRA or on-the-fly workflows, 
                                                      # so you don't override the read values
#"time_overlap_method":0  # don't update time-overlaps - maybe they are already pre-computed and read
#"nac_update_method":0  # just read from computed hvib
#"force_method":0, "rep_force":1 # don't compute forces
#"tsh_method":0 # FSSH
#"hop_acceptance_algo":31 # Boltzman factored probabilities
#"momenta_rescaling_algo":0 # No momentum rescaling

dyn_nbra.update({"ham_update_method":2, "ham_transform_method":0, "time_overlap_method":0, "nac_update_method":0,
                "hvib_update_method":0, "force_method":0, "rep_force":1,
                "tsh_method":0, "hop_acceptance_algo":31, "momenta_rescaling_algo":0,
                "rep_tdse":1, "electronic_integrator":2,
                "decoherence_algo":-1} )

# Explicitly set some defaults (or reasonable value!), to suppress warnings
dyn_nbra.update({"rep_sh":1, "rep_lz":0, "rep_ham":0, "enforce_state_following":0,
                  "enforced_state_indxed":0, "do_phase_correction":0, "do_ssy":0, "phase_correction_tol":1e-10,
                  "max_number_attempts":10,  "total_energy":0.01, "tcnbra_nu_therm":0.01,
                  "thermally_corrected_nbra":0,
                  "tcnbra_nhc_size":1, "tcnbra_do_nac_scaling":1, 
                  "use_Jasper_Truhlar_criterion":1,
                  "fssh3_size_option":0, "fssh3_approach_option":0, "fssh3_decomp_option":0, 
                  "decoherence_C_param":0.1, "decoherence_eps_param":1.0, 
                  "dephasing_informed":0,  "MK_verbosity":0,
                  "sdm_norm_tolerance":1e-4,
                  "use_xf_force":0, "project_out_aux":0, 
                  "ETHD3_alpha":0.01, "ETHD3_beta":0.01, 
                  "Temperature":300.0
                 })

dyn_nbra.update({"prefix":F"fssh_nbra_{model_indx}", 
                 "prefix2":F"fssh_nbra_{model_indx}", })

#####################################
# Select a specific initial condition
icond_indx = args.icond_indx
#####################################  

if method_indx == 1:
    NSTATES = 55
        
    A = MATRIX(NSTATES, NSTATES)
    
    nbatches = 5
    
    for ib in range(nbatches):
        if ib == 0:
            istep, fstep = 0, 1000 + 1
        elif ib == 1:
            istep, fstep = 50, 1050 + 1
        elif ib == 2:
            istep, fstep = 100, 1100 + 1
        elif ib == 3:
            istep, fstep = 150, 1150 + 1
        elif ib == 4:
            istep, fstep = 198, 1198 + 1
    
        E_batch = model_params_nbra["E"][istep:fstep, :]
    
        e2 = np.zeros((fstep-istep, NSTATES, NSTATES))
        for i in range(NSTATES):
            for j in range(NSTATES):
                de = np.average(E_batch[:,i] - E_batch[:,j]) 
                e2[:,i,j] = (E_batch[:,i] - E_batch[:,j] - de)**2
        ave_e2 = np.average(e2, axis=0) # nstates, nstates
        
        invtaus.append(np.sqrt(5/12*ave_e2))
        
    invtaus = np.array(invtaus)

    A = data_conv.nparray2MATRIX(np.average(invtaus, axis=0))
    A.show_matrix()
    
    dyn_nbra.update({"decoherence_algo": 0, "decoherence_times_type": 0, "decoherence_rates": A})

icond_elec = 1

rep = 1 # adiabatic wfc

istates = []
for i in range(NSTATES):
    istates.append(0.0)        

elec_params = {"verbosity":2, "init_dm_type":0,
               "ndia":NSTATES, "nadi":NSTATES, 
               "rep":rep, "init_type":icond_elec, "istates":istates
              }  

if model_indx == 0:
    if icond_indx==0:
        elec_params["istate"] = 1 # The initial state is set to 1, while sampling is done at ground-state PES
elif model_indx == 1:
    if icond_indx==0:
        elec_params["istate"] = 54
        
# Dummy nucl param
nucl_params = { "ndof":1,
            "q":[0.0]*10,
            "p":[0.0]*10, 
            "mass":[1836.0]*10, 
            "force_constant":[0.0]*10, 
            "q_width":[ 0.0 ]*10,
            "p_width":[ 0.0 ]*10,
            "init_type":3 }    

dyn_params = dict(dyn_nbra)

ibatch = args.ibatch

if ibatch == 0:
    dyn_params.update({"icond":0})
elif ibatch == 1:
    dyn_params.update({"icond":50})
elif ibatch == 2:
    dyn_params.update({"icond":100})
elif ibatch == 3:
    dyn_params.update({"icond":150})
elif ibatch == 4:
    dyn_params.update({"icond":198})

if method_indx == 0:
    dyn_params.update({"prefix":F"fssh_nbra_{model_indx}_icond_{icond_indx}_batch_{ibatch}", 
                       "prefix2":F"fssh_nbra_{model_indx}_icond_{icond_indx}_batch_{ibatch}"})
elif method_indx == 1:
    dyn_params.update({"prefix":F"msdm_nbra_{model_indx}_icond_{icond_indx}_batch_{ibatch}", 
                       "prefix2":F"msdm_nbra_{model_indx}_icond_{icond_indx}_batch_{ibatch}"})
    
rnd = Random()

# Run NBRA
dyn_params.update({"nsteps":Nt })

res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params_nbra, elec_params, nucl_params, rnd)

if method_indx == 0:
    pref = F"fssh_nbra_{model_indx}_icond_{icond_indx}_batch_{ibatch}"
elif method_indx == 1:
    pref = F"msdm_nbra_{model_indx}_icond_{icond_indx}_batch_{ibatch}"
    
#============ Plotting ==================
plot_params = { "prefix":pref, "filename":"mem_data.hdf", "output_level":3,"colors": colors, "clrs_index": clrs_index,
                "which_trajectories":[0], "which_dofs":[0], "which_adi_states":list(range(NSTATES)), 
                "which_dia_states":list(range(NSTATES)), 
                "frameon":True, "linewidth":1, "dpi":300,
                "axes_label_fontsize":(8,8), "legend_fontsize":8, "axes_fontsize":(8,8), "title_fontsize":8,
                "what_to_plot":[ "se_pop_adi","sh_pop_adi" ], 
                "which_energies":["potential", "kinetic", "total"],
                "save_figures":1, "do_show":1
              }

tsh_dynamics_plot.plot_dynamics(plot_params)

plt.close('all')

