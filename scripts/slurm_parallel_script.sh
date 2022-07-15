#!/bin/bash

#SBATCH --job-name=multi-compute-sparse-iqp
#SBATCH --account=lcrc
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=36
#SBATCH --output=job.out
#SBATCH --error=job.error
#SBATCH --time=00:30:00

# Run My Program
srun ~lslattery/quantum_kernel/scripts/./multi_compute_qk_Sparse_IQP.sh