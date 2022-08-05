#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

sf=(1.5807 1.4586 1.3494 1.2507 1.1605 1.0775 1.0006 0.9291 0.8622 0.7993 0.7401)
dim=(8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
r=(0.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta{4}/alpha{5}/gamma0.01/ \
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

sf=(1.6152 1.1943 0.9649 0.8282 0.7412 0.683 0.6424 0.6132 0.5915 0.5751 0.5624 0.5525 0.5445 0.538)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
r=(0.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/alpha_test/beta{4}/alpha{5}/gamma0.05/ \
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

sf=(0.5482 0.4699 0.4143 0.3725 0.3396 0.3131 0.2911 0.2726 0.2567 0.2429 0.2307 0.22 0.2104 0.2018 0.194)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.90)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.90/gamma0.4/ \
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

sf=(0.604 0.5178 0.4565 0.4104 0.3743 0.3451 0.3208 0.3004 0.2829 0.2677 0.2543 0.2425 0.2319 0.2224 0.2138)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.95)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.95/gamma0.4/ \
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

sf=(0.8758 0.7014 0.599 0.5339 0.4901 0.4594 0.4369 0.4201 0.4072 0.397 0.3889 0.3824 0.377 0.3725)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.0/gamma0.1/ \
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

sf=(0.8952 0.7163 0.6103 0.5426 0.4967 0.4643 0.4406 0.4227 0.4089 0.398 0.3893 0.3823 0.3764 0.3716)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.33)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.33/gamma0.1/ \
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

sf=(0.9897 0.7981 0.6787 0.599 0.5429 0.502 0.4711 0.4471 0.4282 0.4129 0.4005 0.3901 0.3814 0.3741)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.66)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.66/gamma0.1/ \
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

sf=(1.5807 1.4586 1.3494 1.2507 1.1605 1.0775 1.0006 0.9291 0.8622 0.7993 0.7401)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.9)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.9/gamma0.1/ \
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

sf=(1.5181 1.1871 0.9953 0.875 0.795 0.7392 0.699 0.669 0.6461 0.6283 0.6141 0.6028 0.5935 0.5858)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(0.95)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r0.95/gamma0.1/ \
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

sf=()
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
r=(1.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/r_test/beta{4}/r1.0/gamma0.1/ \
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

