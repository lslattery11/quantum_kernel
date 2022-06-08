"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC

from quantum_kernel.code.utils import get_dataset,get_quantum_kernel,get_projected_quantum_kernel,self_product
from quantum_kernel.code.feature_maps import HeisenbergZZMultiScale
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
        "--n-trotter", type = int,
        required = True,
        help = "number of trotter steps for the feature maps")
    parser.add_argument(
        "--evo-time", type = float,
        required = True,
        help = "total system evolution time")
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
    parser.add_argument(
        "--decimals", type = int,
        required = False,
        default=None,
        help = "number of decimal points to keep (passed directly to np.round)")
    parser.add_argument(
        "--dataset", type = str,
        required = True,
        choices=['fashion-mnist','kmnist','plasticc'],
        help = "dataset to use")
    parser.add_argument(
        "--projected",type=str,
        required = False,
        choices=['huang_proj',''],
        default='',
        help = "use projected quantum kernel or not?")
    parser.add_argument(
        "--simulation-method", type = str,
        required = False,
        default="statevector",
        help = "simulation method to use (passed directly to qiskit)")
    parser.add_argument(
        "--shots", type = int,
        required = False,
        default=1,
        help = "number of shots to use (passed directly to qiskit)")

    args = parser.parse_args()

    if args.projected != '':
        proj='_'+args.projected

    outpath = Path(args.outpath, f"dim_{args.dataset_dim}{proj}_ntrot_{args.n_trotter}_evo_t_{args.evo_time}_init_{args.init_state}_s_{args.init_state_seed}_l1_{args.lam1}_l2_{args.lam2}_dec_{args.decimals}_{args.dataset}_{args.simulation_method}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200)

    lam0=0
    FeatureMap= HeisenbergZZMultiScale.HZZMultiscaleFeatureMap(args.dataset_dim,args.n_trotter,args.evo_time,args.init_state,args.init_state_seed,lam0,args.lam1,args.lam2)
    scaling_factor=FeatureMap.scaling_factor
    x_train *= scaling_factor 
    x_test *= scaling_factor

    if args.decimals is not None:
        x_train = np.around(x_train, decimals=args.decimals)
        x_test = np.around(x_test, decimals=args.decimals)


    if args.projected=='':
        qkern = get_quantum_kernel(FeatureMap,device='CPU',simulation_method=args.simulation_method,shots=args.shots,batch_size=10)
        qkern_matrix_train = qkern.evaluate(x_vec=x_train)
        rdms=None
    else:
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method=args.simulation_method,shots=args.shots,batch_size=10)
        mq=[[i] for i in range(args.dataset_dim)]
        qkern.set_measured_qubits(mq)
        qkern_matrix_train,rdms = qkern.evaluate(x_vec=x_train,return_rdms=True)


    t1 = time.time()
    K_train_time = t1-t0
    print(f"Done computing K_train in {K_train_time}")

    qkern_matrix_test = qkern.evaluate(x_vec=x_test, y_vec=x_train)
    t2 = time.time()
    K_test_time = t2-t1
    print(f"Done computing K_test in {K_test_time}")

    qsvc = SVC(kernel='precomputed')
    qsvc.fit(qkern_matrix_train, y_train)
    score = qsvc.score(qkern_matrix_test, y_test)
    print(f"Score: {score}")

    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'qkern_matrix_test' : qkern_matrix_test,
            'score' : score,
            'args' : args,
            'K_train_time' : K_train_time,
            'K_test_time' : K_test_time,
            'rdms' : rdms
    }
    
    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")