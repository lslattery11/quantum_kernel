"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import pickle
from pathlib import Path
from sklearn.svm import SVC

from quantum_kernel.code.utils import get_dataset,get_quantum_kernel,get_projected_quantum_kernel,self_product
from quantum_kernel.code.feature_maps import IQP

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
        "--log-scaling-factor", type = float,
        required = True,
        help = "scale all data by this factor")
    parser.add_argument(
        "--log-int-scaling-factor", type=float,
        required=True,
        help = "scale all two qubit interactions by this factor")
    parser.add_argument(
        "--density", type = int,
        required = True,
        help = "number of two qubit connections per qubit in each layer")
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

    args = parser.parse_args()

    scaling_factor=10**(args.log_scaling_factor)
    int_time_scale=10**(args.log_int_scaling_factor)

    if args.projected != '':
        proj='_'+args.projected
    else:
        proj=args.projected

    outpath = Path(args.outpath, f"Sparse_IQP_dim_{args.dataset_dim}{proj}_scales_{scaling_factor}_{int_time_scale}_density_{args.density}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    x_train, x_test, y_train, y_test = get_dataset(args.dataset, args.dataset_dim, 800, 200)

    x_train *= scaling_factor 
    x_test *= scaling_factor

    FeatureMap = IQP.Sparse_IQP(args.dataset_dim,density=args.density,data_map_func=self_product,int_time_scale=int_time_scale)

    if args.projected=='':
        qkern = get_quantum_kernel(FeatureMap,device='CPU',batch_size=50)
        qkern_matrix_train = qkern.evaluate(x_vec=x_train)
        rdms=None

    else:
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method='statevector',batch_size=10)
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