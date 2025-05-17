import os
import h5py
import numpy as np
import scipy.sparse as sp

def extract_data(istep, fstep, itraj, hdf_filename, outdir="res"):
    ###########################################
    #istep = 0    # the first timestep to read
    #fstep = 5000 # the last timestep to read
    #itraj = 0
    #hdf_filename = "adiabatic_md0_icond0/mem_data.hdf"
    ###########################################
    nsteps = fstep - istep
    NSTEPS = nsteps
    print(F"Number of steps = {nsteps}")

    nstates = 1
    with h5py.File(F"{hdf_filename}", 'r') as f:    
        nadi = int(f["hvib_adi/data"].shape[2] )
        nstates = nadi                                                           
    print(F"Number of states = {nstates}")

    #================== Read energies =====================
    with h5py.File(hdf_filename, 'r') as f:
        for timestep in range(istep,fstep):
            x = np.array( f["hvib_adi/data"][timestep, itraj, :, :] )
            y = np.array( f["hvib_adi/data"][timestep, itraj, :, :] )
            St_mat = np.array( f["St/data"][timestep, itraj, :, :].real  )

            sp.save_npz(outdir + F"/Hvib_{timestep}_re.npz", sp.csr_matrix(x.real) )
            sp.save_npz(outdir + F"/Hvib_{timestep}_im.npz", sp.csr_matrix(-y.imag) )
            sp.save_npz(outdir + F"/St_{timestep}_re.npz", sp.csr_matrix(St_mat) )

if not os.path.exists("res"):
    os.mkdir("res")

extract_data(12000000, 36000000 + 1, 0, "adiabatic_md0_icond0/mem_data.hdf")
