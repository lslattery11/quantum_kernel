#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/

for i in 0.001 0.002 0.0025 0.03 0.0035 0.004 0.005 0.01 0.1 0.3 0.5 1.0
do
python compute_classical_kernel.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/Laplacian/plasticc/ \
--dataset-dim 18 \
--kernel laplacian \
--gamma $i \
--dataset plasticc
done
