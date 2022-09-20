#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
gamma=(0.0001 0.0002 0.00025 0.0003 0.0004 0.0005 0.001 0.002 0.0025 0.003 0.0035 0.004 0.005 0.01 0.03 0.05 0.1 0.2 0.3 0.4 0.5 1.0)
seed=(42 100 200 3214 10)

parallel \
    --jobs 10 \
    """
        python compute_classical_kernel.py --outpath /media/HomeData/lslattery/QK_project/results/HZZ_multi/plasticc/rbf_test/ \
        --dataset-dim {1} \
        --gamma {2} \
        --dataset plasticc \
        --seed {3} \

    """ ::: "${dim[@]}" ::: "${gamma[@]}" ::: "${seed[@]}"