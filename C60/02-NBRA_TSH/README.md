## Files

* `run_C60.py` - A Python running script to conduct NBRA-TSH dynamics on the C60

* `run_all_nbra.sh` - Distribute the NBRA-TSH dynamics with respect to the method and batch parameters.

* `submit_template_nbra.slm` - Slurm batch file used in the job distribution 

## Note

* The output directory will be `"{method_name}" + _nbra_1_icond_0_batch_X`. `method_name` is either `fssh` or `msdm`. `X` specifies a batch index (5 batches in total).
