#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/
export PYTHONPATH=${PYTHONPATH}:/home/lslattery/

# lsf=$(seq -2.0 0.2 -0.2)

# parallel \
#     --jobs 2 \
#     """
#         python compute_quantum_kernel_Sparse_IQP.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/hproj_Sparse_IQP/plasticc/ \
#         --dataset-dim 18 \
#         --log-scaling-factor {1} \
#         --log-int-scaling-factor 0.0 \
#         --density 17 \
#         --dataset plasticc \
#         --projected huang_proj
#     """ ::: "${lsf[@]}"

sf=(0.01 0.025 0.05 0.1 0.25 0.5)
dim=(4 8 12 16)
parallel \
    --jobs 14 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath ~/QK_project/results/Sparse_IQP/gennorm/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density 0 \
        --beta 1.0 \
        --projected 
    """ ::: "${sf[@]}" ::: "${dim[@]}"