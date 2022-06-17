#!/bin/bash
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/Code/
export PYTHONPATH=${PYTHONPATH}:/mnt/c/Users/lslat/QiskitProjects/VariationalWavefunction/
export PYTHONPATH=${PYTHONPATH}:/media/HomeData/lslattery/


python compute_quantum_kernel_shape.py --outpath /mnt/c/Users/lslat/Data/QK_project/results/kernel_shapes/ \
--dataset-dim 18 \
--log-scaling-factor -0.8 \
--log-int-scaling-factor 0.0 \
--density 17 \
--projected huang_proj