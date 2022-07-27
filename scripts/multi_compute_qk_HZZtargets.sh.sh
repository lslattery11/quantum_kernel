#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/


sf=(0.3753 0.3236 0.2867 0.2588 0.2369 0.2191 0.2043 0.1918 0.181 0.1716 0.1634 0.1561 0.1495 0.1436 0.1383)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta{4}/alpha{5}/gamma0.5/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}"

sf=(0.4667 0.3997 0.3522 0.3164 0.2884 0.2657 0.247 0.2311 0.2176 0.2058 0.1955 0.1863 0.1782 0.1708 0.1642)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta{4}/alpha{5}/gamma0.4/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}"

sf=(0.5971 0.5046 0.4398 0.3915 0.354 0.3239 0.2992 0.2784 0.2607 0.2455 0.2321 0.2204 0.2099 0.2005 0.1921)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta{4}/alpha{5}/gamma0.3/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}"

sf=(0.7988 0.6659 0.5739 0.5061 0.4539 0.4123 0.3784 0.3501 0.3261 0.3055 0.2876 0.2718 0.2579 0.2455 0.2343)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)

parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta{4}/alpha{5}/gamma0.2/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}"