## Files

* `run_sum_patch.py` - A patch summation script to compute global RPI population.

## Note

* The population on a batch is given in `rpi_params["prefix"]X_ibatchY.dat`, 
where `X` is `rpi_sum_params["npatches"]`. `Y` runs from `0` to `len(rpi_params["iconds"]) - 1`. 

* The final population from all batches is saved into `rpi_sum_params["prefix"]_all.dat`.
