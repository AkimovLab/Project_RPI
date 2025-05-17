## Files

* `distribute_patch_jobs.py` - A Python script to run multiple patch dynamics jobs to a cluster. 

* `run_template.py` - Running script template for the job distribution.

* `submit_template.slm` - Slurm batch file used in the job distribution 

## Note

* Here, you need to repetively run patch dynamics by changing `rpi_params["npatches"]` in `distribute_patch_jobs.py` to control the number of patches (20, 200, 500, 1000 for C60)

* The patch dynamics data will be saved in output directories, whose name follows 

`rpi_params["prefix"] + F"n{len(rpi_params["iconds"])}_ibatch{X}" + F"_n{rpi_params["npatches"]}_ipatch{Y}" + F"_n{rpi_params["nstates"]}_istate{Z}"`

Here, `X`, `Y` and `Z` runs:

`X = 0` to `len(rpi_params["iconds"]) - 1`

`Y = 0` to `rpi_params["npatches"] - 1`

`Z = 0` to `rpi_params["nstates"] - 1`

