#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
for i in $(seq -2.0 0.2 -0.2)
do
##for j in $(seq -2.0 0.5 2.0)
for j in 0.0
do
for k in 17
do
python compute_quantum_kernel_Sparse_IQP.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/Sparse_IQPm/fashion-mnist/ --dataset-dim 18 --log-scaling-factor $i --log-int-scaling-factor $j --density $k --dataset fashion-mnist
done
done
done