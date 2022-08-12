"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import scipy
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVC

from quantum_kernel.code.utils import get_dataset,get_quantum_kernel
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
        "--scaling-factor", type = float,
        required = True,
        help = "scale all data by this factor")
    parser.add_argument(
        "--int-scaling-factor", type=float,
        required=True,
        help = "scale all two qubit data interactions by this factor")
    parser.add_argument(
        "--non-data-int-scaling-factor", type=float,
        required=True,
        help = "scale all two qubit non-data interactions by this factor")
    parser.add_argument(
        "--h-layer", type=int,
        required=True,
        choices=[0,1],
        help = "include hadmard layer or not")
    parser.add_argument(
        "--alpha", type = float,
        required = True,
        help = "alpha exponent value for IQP")
    parser.add_argument(
        "--n_trotter", type = int,
        required = True,
        help = "number of trotter steps"
    )
    args = parser.parse_args()

    scaling_factor=args.scaling_factor
    int_scaling_factor=args.int_scaling_factor
    non_data_int_scaling_factor=args.non_data_int_scaling_factor

    #rescale by alpha factor
    int_scaling_factor=int_scaling_factor**(args.alpha/2)

    h_layer=args.h_layer
    if args.projected != '':
        proj='_'+args.projected
    else:
        proj=args.projected

    outpath = Path(args.outpath, f"HZZ_Multi_dim_{args.dataset_dim}_scales_{scaling_factor}_{int_scaling_factor}_{non_data_int_scaling_factor}_hlayer_{h_layer}_alpha_{args.alpha}.p")

    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200)



    n_trotter=args.n_trotter
    init_state='zero'
    init_state_seed = 0
    FeatureMap = HeisenbergZZMultiScale.HZZMultiscaleFeatureMap(args.dataset_dim,n_trotter,init_state,init_state_seed,scaling_factor,int_scaling_factor,non_data_int_scaling_factor,h_layer)

    qkern = get_quantum_kernel(FeatureMap,device='CPU',batch_size=50)
    qkern_matrix_train = qkern.evaluate(x_vec=x_train)

    t1 = time.time()
    K_train_time = t1-t0
    print(f"Done computing K_train in {K_train_time}")

    t0 = time.time()

    qkern_matrix_test = qkern.evaluate(x_vec=x_test, y_vec=x_train)

    K_test_time = t1-t0
    print(f"Done computing K_test in {K_test_time}")
    t1 = time.time()

    qsvc = SVC(kernel='precomputed')
    qsvc.fit(qkern_matrix_train, y_train)
    score = qsvc.score(qkern_matrix_test, y_test)
    print(f"Score: {score}")

    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'qkern_matrix_test' : qkern_matrix_test,
            'args' : args,
            'K_train_time' : K_train_time,
    }

    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")