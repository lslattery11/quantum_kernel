#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/

# lsf=(-1.2 -0.8 -0.4 0.0)

# parallel \
#     --jobs 4 \
#     """
#     python compute_quantum_kernel_shape.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/kernel_shapes/ \
#     --dataset-dim 18 \
#     --log-scaling-factor {1} \
#     --log-int-scaling-factor 0.0 \
#     --density 17 \
#     --projected huang_proj
#     """ ::: "${lsf[@]}"

lam=(0.01 0.025 0.05 0.1)

parallel \
    --jobs 4 \
    """
        python compute_multiscale_quantum_kernel_shape.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/kernel_shapes/ \
        --dataset-dim 14 \
        --n-trotter 10 \
        --evo-time 1 \
        --init-state 'Haar_random' \
        --init-state-seed 0 \
        --lam1 0.01 \
        --lam2 {1} \
        --projected huang_proj
    """ ::: "${lam[@]}"