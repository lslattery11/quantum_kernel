"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import pickle
from pathlib import Path
import numpy as np

from quantum_kernel.code.utils import get_dataset,get_quantum_kernel,get_projected_quantum_kernel,self_product
from quantum_kernel.code.feature_maps import HeisenbergZZMultiScale
from quantum_kernel.code.kernel_measures.generate_data import quantum_pauli_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath", type = Path,
        required = True,
        help = "folder to dump the result")
    parser.add_argument(
        "--dataset-dim", type = int,
        required = True,
        help = "dimensionality (number of qubits)")
    parser.add_argument(
        "--n-trotter", type =int,
        required = True,
        help = "number of trotter steps")
    parser.add_argument(
        "--evo-time", type=float,
        required=True,
        help = "scale all two qubit interactions by this factor")
    parser.add_argument(
        "--init-state", type = str,
        required = True,
        help = "initial state for the Hamiltonian evolution feature map")
    parser.add_argument(
        "--init-state-seed", type = int,
        required = True,
        help = "the seed used for the random initial state")
    parser.add_argument(
        "--lam0", type = float,
        required = True,
        help = "data kinetic term scale")
    parser.add_argument(
        "--lam1", type = float,
        required = True,
        help = "non-data interaction term scale")
    parser.add_argument(
        "--lam2", type = float,
        required = True,
        help = "data interaction term scale")
    parser.add_argument(
        "--n_points", type = int,
        required = True,
        help = "number of data points to generate")
    parser.add_argument(
        "--noise", type = float,
        required = True,
        help = "gaussian noise to data points")
    
    args = parser.parse_args()


    outpath = Path(args.outpath, f"MultiScale_{args.dataset_dim}_ntrot_{args.n_trotter}_evo_t_{args.evo_time}_init_{args.init_state}_s_{args.init_state_seed}_l0_{0}_l1_{args.lam1}_l2_{args.lam2}_n_points_{args.n_points}_noise_{args.noise}_single_z.p")


    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    FeatureMap = HeisenbergZZMultiScale.HZZMultiscaleFeatureMap(args.dataset_dim,args.n_trotter,args.evo_time,args.init_state,args.init_state_seed,args.lam0,args.lam1,args.lam2)
    
    n_points=args.n_points
    seed = 0
    regression = True
    e=args.noise
    rdm_qubits  = [[0]]
    operators = ['Z']

    x_vec,y_vec = quantum_pauli_data(FeatureMap,n_points,seed=seed,regression=regression,e=e,rdm_qubits=rdm_qubits,operators=operators)

    res = {
            'args' : args,
            'seed' : seed,
            'regression' : regression,
            'rdm_qubits' : rdm_qubits,
            'operators' : operators,
            'x_vec' : x_vec,
            'y_vec' : y_vec,
    }

    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")