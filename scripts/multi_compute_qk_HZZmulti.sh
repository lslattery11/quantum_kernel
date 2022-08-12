#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/


mapfile -t sf < logspace-1.5.0.5.20.txt
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16,17,18)
h=(0 1)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_plasticc_data.py --outpath /media/HomeData/lslattery/QK_project/results/HZZ_multi/plasticc/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer {3} \
        --alpha 2.0 \
        --n_trotter 2 \

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${h[@]}"