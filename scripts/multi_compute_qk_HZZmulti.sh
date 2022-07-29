#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/


mapfile -t sf < logspace-1.5.0.5.20.txt
#dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
dim=(2 3 19 20)
seed=(10 1200 33 4 210)
beta=(1.0)
ndisf=(0.0 0.01 0.1 1.0)


parallel \
    --jobs 30 \
    """
        python compute_HZZMultiScale_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/HZZ_multi/final_gennorm/non_data_layer_test/beta{4}/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 0 \
        --non-data-int-scaling-factor {5} \
        --h-layer 0 \
        --alpha 2.0 \
        --beta {4} \
        --seed {3} \
        --n_trotter 4 \
        --projected ''

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${seed[@]}" ::: "${beta[@]}" ::: "${ndisf[@]}"