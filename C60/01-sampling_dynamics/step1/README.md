## Files

* `C60-pos-1.xyz` - a C60 trajectory from the dynamics

* `aligned-C60-pos-1.xyz` - Postprocessed trajectory where translational and rotational motions are removed

* `align_md.py` - Align the MD trajectory by removing the translations and rotations

* `md.inp` - The CP2K input for the ground-state dynamics

* `c60.xyz` - CP2K input geometry

* `BASIS_MOLOPT` - CP2K basis set file

* `GTH_POTENTIALS` - CP2K pseudopotential

* `submit.slm` - Slurm batch file for running a CP2K job

## Note

* Run the CP2K calculation to obtain a trajectory, and align that trajectory with `align_md.py`

* The aligned trajectory will serve as a guiding trajectory in futher NBRA calculations
