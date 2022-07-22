#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

mapfile -t sf < logspace-1.5.0.5.20.txt
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
alpha=(0.5 1 2)
seed=(10 1200 33 4 210)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta1.0/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer True \
        --alpha {3} \
        --beta 1.0 \
        --seed {4} \
        --projected ''

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${alpha[@]}" ::: "${seed[@]}"
