#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/

lam1=(0 0.01 0.025 0.05 0.1 0.25 0.5 1.0)
#lam2=(0 0.01 0.025 0.05 0.1 0.25 0.5 1.0)
lam2=(0.025)
noise=(0.01)

parallel \
    --jobs 2 \
    """
        python generate_quantum_pauli_data.py --outpath /mnt/c/Users/lslat/Data/QK_project/quantum_data/MultiScale/single_z \
        --dataset-dim 8 \
        --n-trotter 10 \
        --evo-time 1 \
        --init-state 'Haar_random' \
        --init-state-seed 0 \
        --lam0 0 \
        --lam1 {1} \
        --lam2 {2} \
        --n_points 1000 \
        --noise {3}
    """ ::: "${lam1[@]}" ::: "${lam2[@]}" ::: "${noise[@]}"