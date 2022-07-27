#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/


sf=(2.0739)
dim=(12)
seed=(10 1200 33 4 210)
beta=(0.1)
alpha=(2.0)
size=(50 100 200 500 1000 2000)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/matrix_size_test/beta{4}/alpha{5}/gamma0.3/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"

sf=(0.2607)
dim=(12)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
size=(50 100 200 500 1000 2000)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/matrix_size_test/beta{4}/alpha{5}/gamma0.3/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"

sf=(0.2531)
dim=(12)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
size=(50 100 200 500 1000 2000)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/matrix_size_test/beta{4}/alpha{5}/gamma0.3/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 1 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"

mapfile -t sf < logspace-2.1.30.txt
dim=(12)
seed=(10 1200 33 4 210)
beta=(0.1)
alpha=(2.0)
size=(50 100 200 500 1000 2000)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/matrix_size_test/beta{4}/alpha{5}/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 0 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"

mapfile -t sf < logspace-2.1.30.txt
dim=(12)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
size=(50 100 200 500 1000 2000)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/matrix_size_test/beta{4}/alpha{5}/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 0 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"

mapfile -t sf < logspace-2.1.30.txt
dim=(12)
seed=(10 1200 33 4 210)
beta=(2.0)
alpha=(2.0)
size=(50 100 200 500 1000 2000)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/matrix_size_test/beta{4}/alpha{5}/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0.0 \
        --h-layer 0 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 2 \
        --matrix_size {6} \
        --projected ''

    """ ::: "${sf[@]}" :::+ "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"