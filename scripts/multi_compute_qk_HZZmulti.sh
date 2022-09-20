#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

# sf=(0.01 0.0115 0.0134 0.0156 0.0181 0.0209 0.0242 0.0281 0.0326 0.0378 0.0437 0.0507 0.0571 0.0642 0.0723 0.0813 0.0915 0.103 0.1159 0.1304 0.1468 0.1652 0.1859 0.2092)
# dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
# h=(0 1)
# ms=(700)
# seed=(0 1 2 3)

# parallel \
#     --jobs 24 \
#     """
#         python compute_qk_HZZMultiScale_plasticc_data.py --outpath /media/HomeData/lslattery/QK_project/results/HZZ_multi/fashion-mnist/dim_test/ \
#         --dataset-dim {2} \
#         --scaling-factor {1} \
#         --int-scaling-factor {1} \
#         --non-data-int-scaling-factor 0 \
#         --h-layer {3} \
#         --alpha 2.0 \
#         --n_trotter 2 \
#         --matrix_size {4} \
#         --split_index {5} \
#     """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${h[@]}" ::: "${ms[@]}" ::: "${seed[@]}"


sf=(0.0242 0.0281 0.0326 0.0378 0.0437 0.0507 0.0571 0.0642 0.0723 0.0813 0.0915 0.103 0.1159 0.1304 0.1468 0.1652 0.1859 0.2092)
dim=(16)
h=(0)
ms=(50 100 200 300 400 500 600 700 1000 1500 2000 2500)
seed=(0 1 2 3)

parallel \
    --jobs 24 \
    """
        python compute_qk_HZZMultiScale_plasticc_data.py --outpath /media/HomeData/lslattery/QK_project/results/HZZ_multi/fashion-mnist/matrix_size_test2.0/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer {3} \
        --alpha 2.0 \
        --n_trotter 2 \
        --matrix_size {4} \
        --split_index {5} \
    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${h[@]}" ::: "${ms[@]}" ::: "${seed[@]}"

sf=(0.074)
dim=(16)
h=(1)
ms=(50 100 200 300 400 500 600 700 1000 1500 2000 2500)
seed=(0 1 2 3)

parallel \
    --jobs 24 \
    """
        python compute_qk_HZZMultiScale_plasticc_data.py --outpath /media/HomeData/lslattery/QK_project/results/HZZ_multi/fashion-mnist/matrix_size_test2.0/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer {3} \
        --alpha 2.0 \
        --n_trotter 2 \
        --matrix_size {4} \
        --split_index {5} \
    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${h[@]}" ::: "${ms[@]}" ::: "${seed[@]}"

