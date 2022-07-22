#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

mapfile -t sf < logspace-1.5.0.5.20.txt
ndisf=(0.001 0.01 0.1)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
seed=(10 1200 33 4 210)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/a/beta1.0/ \
        --dataset-dim {3} \
        --scaling-factor {1} \
        --int-scaling-factor 0.0 \
        --non-data-int-scaling-factor {2} \
        --h-layer True \
        --beta 1.0 \
        --seed {4} \
        --projected ''
    """ ::: "${sf[@]}" ::: "${ndisf[@]}" ::: "${dim[@]}" ::: "${seed[@]}"
