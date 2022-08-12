#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/


sf=(0.0316 0.0356 0.04 0.0451 0.0507 0.0571 0.0642 0.0723 0.0813 0.0915 0.103 0.1159 0.1304 0.1468 0.1652 0.1859 0.2092 0.2354 0.2649 0.2981 0.3355 0.3775 0.4248 0.4781 0.538 0.6054 0.6813 0.7667 0.8628 0.9709 1.0926 1.2295 1.3836 1.5571 1.7522 1.9718 2.219 2.4971 2.8101 3.1623)
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18)
h=(0 1)

parallel \
    --jobs 10 \
    """
        python compute_qk_HZZMultiScale_plasticc_data.py --outpath /media/HomeData/lslattery/QK_project/results/HZZ_multi/plasticc/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor {1} \
        --non-data-int-scaling-factor 0 \
        --h-layer {3} \
        --alpha 2.0 \
        --n_trotter 2 \

    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${h[@]}"