#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
for i in $(seq -2.0 0.2 -0.2)
do
python compute_quantum_kernel_Sparse_IQP.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/hproj_Sparse_IQPm/plasticc/ --dataset-dim 14 \
--log-scaling-factor $i \
--log-int-scaling-factor 0 \
--density 13 \
--dataset plasticc \
--projected huang_proj
done