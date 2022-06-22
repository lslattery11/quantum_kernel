#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

lsf=(-1.2 -0.8 -0.4 0.0 -1.2 -0.8 -0.4 0.0)
lisf=(0 0 0 0 -2.0 -2.0 -2.0 -2.0)

parallel --link \
    --jobs 8 \
    """
    python compute_quantum_kernel_shape.py --outpath ~/QK_project/results/kernel_shapes/ \
    --dataset-dim 14 \
    --log-scaling-factor {1} \
    --log-int-scaling-factor {2} \
    --density 13 \
    --projected 'single_proj'
    """ ::: "${lsf[@]}" ::: "${lisf[@]}"

#lam1=(0.01 0.025 0.05 0.1 0.01 0.01 0.01 0.01 0.01)
#lam2=(0.01 0.01 0.01 0.01 0.025 0.05 0.1 0.25 1.0)
lam1=(0.01 0.025 0.05 0.1)
lam2=(0.025 0.025 0.025 0.025)
parallel --link \
    --jobs 7 \
    """
        python compute_multiscale_quantum_kernel_shape.py --outpath ~/QK_project/results/kernel_shapes/ \
        --dataset-dim 14 \
        --n-trotter 10 \
        --evo-time 1 \
        --init-state 'Haar_random' \
        --init-state-seed 0 \
        --lam1 {1} \
        --lam2 {2} \
        --projected huang_proj
    """ ::: "${lam1[@]}" ::: "${lam2[@]}"