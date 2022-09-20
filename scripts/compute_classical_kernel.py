"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.neighbors import KNeighborsTransformer
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel, laplacian_kernel,cosine_similarity
from sklearn.metrics import balanced_accuracy_score
from quantum_kernel.code.utils import get_dataset,precomputed_kernel_GridSearchCV

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath", type = Path,
        required = True,
        help = "folder to dump the result")
    parser.add_argument(
        "--dataset-dim", type = int,
        required = True,
        help = "dimensionality")
    parser.add_argument(
        "--gamma", type = float,
        required = True,
        help = "kernel hyper parameter")
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
        "--seed", type = int,
        required = True,
        help = "seed")

    args = parser.parse_args()

    kernel_dict={'rbf':rbf_kernel}

    outpath = Path(args.outpath, f"rbf_dim_{args.dataset_dim}_gamma_{args.gamma}_dec_{args.decimals}_{args.dataset}_{args.seed}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")


    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200,args.seed)

    #x_train *= args.gamma
    #x_test *= args.gamma

    if args.decimals is not None:
        x_train = np.around(x_train, decimals=args.decimals)
        x_test = np.around(x_test, decimals=args.decimals)

    kernel_function=kernel_dict['rbf']

    kern_matrix_train = kernel_function(x_train,gamma=args.gamma)

    t1 = time.time()
    K_train_time = t1-t0
    print(f"Done computing K_train in {K_train_time}")

    kern_matrix_test = kernel_function(x_test,x_train,gamma=args.gamma)
    
    t2 = time.time()
    K_test_time = t2-t1
    print(f"Done computing K_test in {K_test_time}")

    qsvc = SVC(kernel='precomputed')
    qsvc.fit(kern_matrix_train, y_train)
    score = qsvc.score(kern_matrix_test, y_test)
    print(f"Score: {score}")

    Cs=[0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024]
    best_C,best_g=precomputed_kernel_GridSearchCV(kern_matrix_train,y_train,Cs,[0])

    qsvc = SVC(kernel='precomputed', C=best_C)
    qsvc.fit(kern_matrix_train, y_train)
    y_pred_train = qsvc.predict(kern_matrix_train)

    train_score=balanced_accuracy_score(y_train,y_pred_train)

    y_pred_test = qsvc.predict(kern_matrix_test)
    test_score = balanced_accuracy_score(y_test,y_pred_test)

    print(f"Optimized Train Score: {train_score}, and test score: {test_score}")

    res = {
            'qkern_matrix_train' : kern_matrix_train,
            'qkern_matrix_test' : kern_matrix_test,
            'args' : args,
            'K_train_time' : K_train_time,
            'K_test_time' : K_test_time,
            'tuned_train_score' : train_score,
            'tuned_test_score' : test_score,
    }
    
    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")