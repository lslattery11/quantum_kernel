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

#sf=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
mapfile -t sf < logspace-2.1.30.txt
dim=(4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
beta=(0.1 1.0 2.0)
seed=(10 1200 33 4 210)
parallel \
    --jobs 14 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/Sparse_IQP/final_gennorm/\
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density 0 \
        --beta {3} \
        --seed {4} \
        --projected ''
    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${beta[@]}" ::: "${seed[@]}"

#calculated from q_curve interpolation
parallel \
    --jobs 14 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath /nfs/gce/projects/gce/QK_project/results/Sparse_IQP/final_gennorm/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density {2} \
        --beta {3} \
        --seed {4} \
        --projected ''
    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${beta[@]}" ::: "${seed[@]}"