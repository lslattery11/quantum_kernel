#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/

for i in 0.02 0.05 0.1
do
for j in 1 2 3
do
for k in 0
do
python compute_quantum_kernel_Heisenberg1D.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/Heisenberg1D/kmnist/ --dataset-dim 17 \
--n-trotter 40 \
--evo-time $i \
--init-state 'Haar_random' \
--init-state-seed 0 \
--k $j \
--r $k \
--dataset kmnist
done
done
done