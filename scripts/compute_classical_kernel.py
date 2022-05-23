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
from sklearn.metrics.pairwise import rbf_kernel,cosine_similarity
from quantum_kernel.code.utils import get_dataset
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
        "--kernel", type=str,
        required=True,
        help = "type of kernel (rbf,cosine ..etc)")
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

    args = parser.parse_args()

    kernel_dict={'rbf':rbf_kernel}

    outpath = Path(args.outpath, f"dim_{args.dataset_dim}_kernel_{args.kernel}_gamma_{args.gamma}_dec_{args.decimals}_{args.dataset}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")


    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200)

    #x_train *= args.gamma
    #x_test *= args.gamma

    if args.decimals is not None:
        x_train = np.around(x_train, decimals=args.decimals)
        x_test = np.around(x_test, decimals=args.decimals)

    kernel_function=kernel_dict[args.kernel]

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

    res = {
            'qkern_matrix_train' : kern_matrix_train,
            'qkern_matrix_test' : kern_matrix_test,
            'score' : score,
            'args' : args,
            'K_train_time' : K_train_time,
            'K_test_time' : K_test_time,
    }
    
    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")