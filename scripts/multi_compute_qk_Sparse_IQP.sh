#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/

lsf=$(seq -2.0 0.2 -0.2)

parallel \
    --jobs 17 \
    """
        python compute_quantum_kernel_Sparse_IQP.py --outpath /media/HomeData/lslattery/QK_project/results/hproj_Sparse_IQP/fashion-mnist/ \
        --dataset-dim 18 \
        --log-scaling-factor {1} \
        --log-int-scaling-factor 0.0 \
        --density 17 \
        --dataset fashion-mnist \
        --projected huang_proj
    """ ::: "${lsf[@]}"