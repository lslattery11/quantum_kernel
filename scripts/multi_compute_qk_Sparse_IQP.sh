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

#sf=(0.001 0.0025 0.005 0.01 0.025 0.05 0.1 0.25 0.5 1.0)
sf=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)
#beta=(0.05 0.1 0.2 0.5 1.0 5.0)
beta=(0.1 1.0 2.0)

parallel \
    --jobs 14 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath ~/QK_project/results/Sparse_IQP/gennorm/gamma0.2/ \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density 0 \
        --beta {3} \
        --projected ''
    """ ::: "${sf[@]}" ::: "${dim[@]}" ::: "${beta[@]}"

#calculated from q_curve interpolation
sf=(0.6745147380670223 0.5684518372842262 0.49476028790019533 0.44998769698234725 0.4052151060644992 0.3813693660751997 0.36028254130998344 0.3391957165447671 0.3181088917795508 0.29770041798863284 0.2814170149905274 0.2651336119924219 0.24885020899431642 0.23256680599621096])
dim=(5 6 7 8 9 10 11 12 13 14 15 16 17 18)
parallel --link\
    --jobs 14 \
    """
        python compute_Sparse_IQP_kernel_gennorm_data.py --outpath ~/QK_project/results/Sparse_IQP/gennorm/gamma0.2 \
        --dataset-dim {2} \
        --scaling-factor {1} \
        --int-scaling-factor 1.0 \
        --density {2} \
        --beta {3} \
        --projected ''
    """ ::: "${sf[@]}" ::: "${dim[@]}"