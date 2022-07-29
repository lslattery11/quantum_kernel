#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

sf=(0.4697 0.3767 0.3146 0.2702 0.2368 0.2107 0.1899 0.1728 0.1586 0.1465 0.1362 0.1272 0.1193 0.1124 0.1062)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
ndisf=(1.0)
parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/non_data_layer_test/beta{4}/gamma0.3/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 0 \
        --non-data-int-scaling-factor {6} \
        --h-layer 0 \
        --alpha {5} \
        --beta {4} \
        --seed {3} \
        --n_trotter 4 \
        --matrix_size 500 \
        --projected ''

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${ndisf[@]}"

sf=(0.2607)
dim=(12)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
size=(50 100 200 500 1000 2000 5000 10000)
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

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"

mapfile -t sf < logspace-2.1.30.txt
dim=(12)
seed=(10 1200 33 4 210)
beta=(1.0)
alpha=(2.0)
size=(50 100 200 500 1000 2000 5000 10000)
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

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${alpha[@]}" ::: "${size[@]}"


