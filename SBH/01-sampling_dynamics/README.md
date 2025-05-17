## Files

* `run_sampling.py` - A Python running script to conduct ground-state adiabatic dynamics on the SBH

* `initial_conditions.json` - Wigner sampling information

* `extract_HS.py` - Extract vibronic Hamiltonian and time overlap data and save them as the npz format 

## Note

* First, run `run_sampling.py`. Then you will have the ground-state dynamics result. Then running `extract_HS.py` will give vibronic Hamiltonian and time overlap data in `res/`
