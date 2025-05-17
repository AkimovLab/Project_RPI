## Files

* `aligned-C60-pos-1.xyz` - The NBRA trajectory from step1

* `clean.sh` - Clean output data 

* `distribute_jobs.py` - Distribute the time overlap calculation jobs

* `es_diag_temp.inp` - The CP2K TDDFT input template

* `run_template.py` - A running template script to conduct time overlap calculation, used in `distribute_jobs.py`

* `submit_template.slm` - Slurm batch file for executing the running script, used in `distribute_jobs.py`


## Note

* Distribute the timeoverlap calculations with `distribute_jobs.py`. Libra interface will run TDDFT calculations with CP2K along with the given aligned trajectory. 

