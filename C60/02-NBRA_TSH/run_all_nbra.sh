#!/bin/bash
for j in 0; do
    for i in $(seq 0 4); do
        sleep 1m
        echo "Submitting jobs for ibatch=$i and method=$j ##############"
        sed -i "s/python.*/python Run_NBRA.py --method_indx $j --icond_indx 0 --param_indx 0 --ibatch $i > log$i/g" submit_template_nbra.slm
        sbatch submit_template_nbra.slm
    done
done
