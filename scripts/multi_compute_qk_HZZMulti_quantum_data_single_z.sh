#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/

lam1=(0 0.01 0.025 0.05 0.1 0.25 0.5)
lam2=(0.01 0.025 0.05 0.1 0.25 0.5)
target_lam1=(0 0.01 0.025 0.05)
parallel \
    --jobs 6 \
    """
        python compute_qk_HZZMultiScale_quantum_data_single_z.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/MultiScale/single_z \
        --datapath /mnt/c/Users/lslat/Data/QK_project/quantum_data/MultiScale/single_z \
        --dataset-dim 8 \
        --n-trotter 10 \
        --evo-time 1 \
        --init-state 'Haar_random' \
        --init-state-seed 0 \
        --lam0 0 \
        --lam1 {1} \
        --lam2 {2} \
        --target_lam0 0 \
        --target_lam1 {3} \
        --target_lam2 0.025 \
        --noise 0.01 \
        --projected single_proj \
    """ ::: "${lam1[@]}" ::: "${lam2[@]}" ::: "${target_lam1[@]}"