#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

sf=(0.4156 0.3625 0.3243 0.2951 0.2719 0.253 0.2372 0.2238 0.2122 0.202 0.1931 0.1851 0.1779 0.1714 0.1655)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.0/gamma0.4/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size 500 \
        --r {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${r[@]}"

sf=(0.4225 0.368 0.3288 0.2989 0.2752 0.2558 0.2397 0.226 0.2141 0.2038 0.1947 0.1865 0.1792 0.1726 0.1666)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.33)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.33/gamma0.4/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size 500 \
        --r {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${r[@]}"

sf=(0.4641 0.4003 0.3547 0.3203 0.2932 0.2712 0.2529 0.2374 0.2241 0.2125 0.2023 0.1933 0.1852 0.1779 0.1713)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.66)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.66/gamma0.4/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size 500 \
        --r {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${r[@]}"

sf=(0.7896 0.6844 0.6089 0.5516 0.5064 0.4696 0.4389 0.4129 0.3905 0.371 0.3538 0.3385 0.3248 0.3124 0.3012)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(1.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r1.0/gamma0.4/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size 500 \
        --r {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${r[@]}"