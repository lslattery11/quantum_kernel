#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

#mapfile -t sf < logspace-2.1.30.txt
sf=(0.566 0.506 0.449 0.417 0.384 0.354 0.338 0.322 0.306 0.29 0.278 0.27 0.262 0.255)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
seed=(10 1200 33 4 210)

parallel \
    --jobs 15 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/Sparse_IQP/final_gennorm/beta1.0gamma0.2/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density {2} \
        --beta 1.0 \
        --seed {3} \
        --projected ''
    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}"