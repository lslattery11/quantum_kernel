#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/


mapfile -t sf < logspace-1.5.0.5.20.txt
dim=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
r=(0.9 0.95)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r{5}/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer 1 \
        --alpha 2.0 \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size 500 \
        --r {5} \
        --projected ''

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${r[@]}"

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r{5}/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer 0 \
        --alpha 2.0 \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size 500 \
        --r {5} \
        --projected ''

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${r[@]}"