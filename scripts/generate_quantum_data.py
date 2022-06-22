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
        "--lam1", type = float,
        required = True,
        help = "non-data interaction term scale")
    parser.add_argument(
        "--lam2", type = float,
        required = True,
        help = "data interaction term scale")

    
    args = parser.parse_args()

    if args.projected != '':
        proj='_'+args.projected
    else:
        proj=args.projected

    outpath = Path(args.outpath, f"MultiScale_{args.dataset_dim}{proj}_ntrot_{args.n_trotter}_evo_t_{args.evo_time}_init_{args.init_state}_s_{args.init_state_seed}_l1_{args.lam1}_l2_{args.lam2}_dec_{args.decimals}_{args.simulation_method}_shape_test.p")


    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    y_vec = np.random.rand(1,args.dataset_dim)

    x_train = np.random.rand(1000,args.dataset_dim)
    #scale vectors so that we get a flat distribution of x_train sizes.
    x_train_norms = np.linalg.norm(x_train,axis=1)
    flat_norms=np.linspace(0.01,5,num=1000)
    x_train = (flat_norms/x_train_norms*x_train.T).T

    #flat_norms=np.linspace(1,5,num=1000).reshape(1000,1)
    #x_train = flat_norms*y_vec

    norm_diffs=np.linalg.norm(y_vec-x_train,axis=1)


    FeatureMap = HeisenbergZZMultiScale.HZZMultiscaleFeatureMap(args.dataset_dim,args.n_trotter,args.evo_time,args.init_state,args.init_state_seed,0,args.lam1,args.lam2)

    if args.projected=='':
        qkern = get_quantum_kernel(FeatureMap,device='CPU',batch_size=50)
        qkern_matrix_train = qkern.evaluate(x_vec=x_train,y_vec=y_vec)
        rdms=None
    elif args.projected=='huang_proj':
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method='statevector',batch_size=10)
        mq=[[i] for i in range(args.dataset_dim)]
        qkern.set_measured_qubits(mq)
        qkern_matrix_train,rdms = qkern.evaluate(x_vec=x_train,y_vec=y_vec,return_rdms=True)

    elif args.projected=='single_proj':
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method='statevector',batch_size=10)
        qkern_matrix_train,rdms = qkern.evaluate(x_vec=x_train,y_vec=y_vec,return_rdms=True)



    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'args' : args,
            'rdms' : rdms,
            'flat_norms': flat_norms,
            'norm_diffs': norm_diffs
    }

    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")