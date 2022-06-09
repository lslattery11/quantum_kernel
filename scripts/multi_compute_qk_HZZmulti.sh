#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/

data_dim=(4 6 8 10 12)
lam1=(0 0.01 0.025 0.1 0.25 1.0)
lam2=(0 0.01 0.025 0.1 0.25 1.0)

parallel \
    --jobs 17 \
    """
        python compute_quantum_kernel_HZZMulti.py --outpath /media/HomeData/lslattery/QK_project/results/hproj_HZZMulti/fashion-mnist/ \
        --dataset-dim {1} \
        --n-trotter 10 \
        --evo-time 1 \
        --init-state 'Haar_random' \
        --init-state-seed 0 \
        --lam1 {2} \
        --lam2 {3} \
        --dataset fashion-mnist \
        --projected huang_proj
    """ ::: "${data_dim[@]}" ::: "${lam1[@]}" ::: "${lam2[@]}"