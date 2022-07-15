#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

mapfile -t sf < logspace-2.1.30.txt
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
beta=(0.1 1.0 2.0)
seed=(10 1200 33 4 210)

parallel \
    --jobs 36 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath /home/lslattery/QK_project/results/Sparse_IQP/final_gennorm/quantum/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density {2} \
        --beta {3} \
        --seed {4} \
        --projected ''
    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${beta[@]}" ::: "${seed[@]}"