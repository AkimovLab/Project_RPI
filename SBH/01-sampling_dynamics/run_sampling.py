#!/usr/bin/env python

import sys
import cmath
import math
import os
import h5py
import json
import matplotlib.pyplot as plt   # plots
import numpy as np
import time
import warnings

if sys.platform=="cygwin":
    from cyglibra_core import *
elif sys.platform=="linux" or sys.platform=="linux2":
    from liblibra_core import *

import util.libutil as comn
from libra_py import units, data_conv #, dynamics_plotting
import libra_py.models.Libra as model_Libra
import libra_py.models.GLVC as GLVC

from libra_py import dynamics_plotting

import libra_py.dynamics.tsh.compute as tsh_dynamics
import libra_py.dynamics.tsh.plot as tsh_dynamics_plot

import libra_py.dynamics.exact.compute as dvr
import libra_py.dynamics.exact.save as dvr_save

# Set the interface function
def compute_model(q, params, full_id):

    model = params["model"]
    res = None

    if model==1:
        res = GLVC.GLVC(q, params, full_id)
    else:
        pass

    return res

def potential(q, params):
    full_id = Py2Cpp_int([0,0]) 
    
    return compute_model(q, params, full_id)

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

all_model_params = [model_params1]

#################################
# Give the model used an index
model_indx = 0
################################

model_params = all_model_params[model_indx]

list_states = [x for x in range(model_params["nstates"])]
NSTATES = model_params["nstates"]

# Do the sampling one more step for further use in the RPI integration
if model_indx == 0:
    Nt = 40000000 + 1 # ~ 100 ps

# ## Trajectory Sampling
dyn_adi = { "nsteps":Nt, "ntraj":1, "nstates":NSTATES,
                "dt":0.1, "num_electronic_substeps":1, "isNBRA":0, "is_nbra":0,
                "progress_frequency":1, "which_adi_states":range(NSTATES), "which_dia_states":range(NSTATES),      
                "mem_output_level":4,
                "properties_to_save":[ "timestep", "time", "q", "p", "f", "Epot_ave", "Ekin_ave", "Etot_ave", 
                                      "hvib_adi", "basis_transform", "St"],
                "prefix":F"adiabatic_md{model_indx}", "prefix2":F"adiabatic_md{model_indx}"
              }

dyn_adi.update({"tsh_method":-1, "ham_update_method":1, "ham_transform_method":1, "time_overlap_method":1 }) 

## Initial conditions
icond_nucl = 3 

nucl_params = { "ndof":1,
                "q":[-10.0],
                "p":[0.0], 
                "mass":[2000.0], 
                "force_constant":[0.000125], 
                "q_width":[ 0.0 ],
                "p_width":[ 0.0 ],
                "init_type":icond_nucl }

icond_elec = 0

rep = 1 # adiabatic wfc

istates = []
for i in range(NSTATES):
    istates.append(0.0)        

elec_params = {"verbosity":2, "init_dm_type":0,
               "ndia":NSTATES, "nadi":NSTATES, 
               "rep":rep, "init_type":icond_elec, "istates":istates
              }

#####################################
# Select a specific initial condition
icond_indx = 0
#####################################    

if model_indx == 0:
    elec_params["istate"] = 0    
        
    # Nuclear initial conditions 
    beta = model_params["beta"]
    _nosc = ndof = model_params["num_osc"]
    _nst = NSTATES 
    
    # The minimum of the position and the Wigner sampling momenta
    q_init = []
    p_init = []
    for i in range(len(freqs)):
        r_i = g_vals[i]/(model_params["mass"][i]*freqs[i]*freqs[i])
        p_i = np.sqrt(model_params["mass"][i]*freqs[i])
        q_init.append(r_i)
        p_init.append(p_i)
    
    # 1/2 * mass * beta * w**2 = 1/2 * (1/sigma_q**2) => sigma_q = 1/(w * sqrt(mass * beta))
    # 1/2 * (beta/mass) = 1/2 * (1/sigma_p**2) => sigma_p = 1/(sqrt(beta/mass))
    # force_constant = mass * w**2
    nucl_params = { "ndof":ndof,
                    #"q":[0.0 for i in range(ndof)],
                    # Average of the position for each dof
                    "q": q_init.copy(),
                    # Average of the momentum for each dof
                    "p": [0.0]*_nosc,
                    "mass":list(model_params["mass"]),
                    "force_constant":[ model_params["mass"][i]*model_params["omega"][0][i]**2 for i in range(_nosc) ],
                    "q_width":[ 0.0 ]*_nosc,
                    "p_width":[ 0.0 ]*_nosc,
                    "init_type": 4
                  }        

# To compute the vertical excitation
def get_Eadi(_model_indx, q):
    nst = all_model_params[_model_indx]["nstates"]
    res = potential(q, all_model_params[_model_indx])
    H = np.zeros((nst, nst), dtype=complex)
    for i in range(nst):
        for j in range(nst):
            H[i,j] = res.ham_dia.get(i,j)
    E, U = np.linalg.eig(H) 
    idx= np.argsort(E) 
    E = E[idx] 
    U = U[:,idx]

    return E

ndof = len(q_init)
q = MATRIX(ndof,1)

for idof in range(ndof):
    q.set(idof, 0, q_init[idof])

E = np.float64(get_Eadi(0, q))

# Find the initial distortion
vert_gap = E[1] - E[0]

## Read the Wigner sampling data provided by S. Mukherjee
with open("initial_conditions.json", "r") as f:
    icond_data = json.load(f)

qs, ps = [], []

m = 1836.0 # 1 amu

for icond in range(len(icond_data)):
    qs.append( np.array(icond_data[icond]["positions"]) )
    ps.append( m*np.array(icond_data[icond]["velocities"]) )

qs = np.array(qs)[..., 0]
ps = np.array(ps)[..., 0]

# Add distortion to consider the configurational space associated with a vertical excitation from the original non-NBRA calculation
opt_indx = 320
q_dist = q_init[4] - np.sqrt((qs[opt_indx, 4] - q_init[4])**2 + 2*vert_gap/nucl_params["force_constant"][4] )

########################
for dof in range(_nosc):
    nucl_params["q"][dof] = qs[opt_indx, dof] 
    nucl_params["p"][dof] = ps[opt_indx, dof] 

nucl_params["q"][4] = q_dist # Give a distortion to the dof = 4, amounting to the vertical transition energy

dyn_params = dict(dyn_adi)
dyn_params.update({"prefix":F"adiabatic_md{model_indx}_icond{icond_indx}", 
                    "prefix2":F"adiabatic_md{model_indx}_icond{icond_indx}"})

rnd = Random()
res = tsh_dynamics.generic_recipe(dyn_params, compute_model, model_params, elec_params, nucl_params, rnd)

