#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/

for i in 0 0.1 1 2
do
for j in 0 0.001 0.1 0.2 0.5 1.0
do
python compute_quantum_kernel_HZZMulti.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/hproj_HZZMulti/fashion-mnist/ --dataset-dim 10 \
--n-trotter 10 \
--evo-time 1 \
--init-state 'Haar_random' \
--init-state-seed 0 \
--lam1 $i \
--lam2 $j \
--dataset fashion-mnist \
--projected huang_proj
done
done