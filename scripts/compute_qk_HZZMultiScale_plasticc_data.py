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
from sklearn.metrics import balanced_accuracy_score

from quantum_kernel.code.utils import get_dataset,get_quantum_kernel,precomputed_kernel_GridSearchCV
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
    parser.add_argument(
        "--matrix_size", type = int,
        required = True,
        help = "number of trotter steps"
    )
    parser.add_argument(
        "--split_index", type = int,
        required = True,
        help = ""
    )
    args = parser.parse_args()

    scaling_factor=args.scaling_factor
    int_scaling_factor=args.int_scaling_factor
    non_data_int_scaling_factor=args.non_data_int_scaling_factor

    #rescale by alpha factor
    int_scaling_factor=int_scaling_factor**(args.alpha/2)

    h_layer=args.h_layer

    outpath = Path(args.outpath, f"HZZ_Multi_dim_{args.dataset_dim}_scales_{scaling_factor}_{int_scaling_factor}_{non_data_int_scaling_factor}_hlayer_{h_layer}_alpha_{args.alpha}_matrix_size_{args.matrix_size}_split_{args.split_index}.p")

    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    if args.matrix_size <= 700:
        x_train, x_test, y_train, y_test = get_dataset("plasticc", args.dataset_dim, 0.8, 0.2,split_index=args.split_index)

        x_train=x_train[0:args.matrix_size]
        y_train=y_train[0:args.matrix_size]
    else:
        x_train, x_test, y_train, y_test = get_dataset("plasticc", args.dataset_dim, args.matrix_size, min(1000,3506-args.matrix_size),split_index=-1)

    ##temp for now.
    x_train, x_test, y_train, y_test = get_dataset("fashion-mnist", args.dataset_dim, args.matrix_size, 1000,split_index=args.split_index)
    ##

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

    t1 = time.time()
    K_test_time = t1-t0
    print(f"Done computing K_test in {K_test_time}")

    qsvc = SVC(kernel='precomputed')
    qsvc.fit(qkern_matrix_train, y_train)
    score = qsvc.score(qkern_matrix_test, y_test)
    print(f"Score: {score}")

    Cs=[0.001, 0.002, 0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024]
    best_C,best_g=precomputed_kernel_GridSearchCV(qkern_matrix_train,y_train,Cs,[0])

    qsvc = SVC(kernel='precomputed', C=best_C)
    qsvc.fit(qkern_matrix_train, y_train)
    y_pred_train = qsvc.predict(qkern_matrix_train)

    train_score=balanced_accuracy_score(y_train,y_pred_train)

    y_pred_test = qsvc.predict(qkern_matrix_test)
    test_score = balanced_accuracy_score(y_test,y_pred_test)

    print(f"Optimized Train Score: {train_score}, and test score: {test_score}")

    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'qkern_matrix_test' : qkern_matrix_test,
            'args' : args,
            'K_train_time' : K_train_time,
            'tuned_train_score' : train_score,
            'tuned_test_score' : test_score,
    }

    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")