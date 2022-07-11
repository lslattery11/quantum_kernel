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

from quantum_kernel.code.utils import get_gennorm_samples,get_quantum_kernel,get_projected_quantum_kernel,self_product
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
        "--scaling-factor", type = float,
        required = True,
        help = "scale all data by this factor")
    parser.add_argument(
        "--int-scaling-factor", type=float,
        required=True,
        help = "scale all two qubit interactions by this factor")
    parser.add_argument(
        "--density", type = int,
        required = True,
        help = "number of two qubit connections per qubit in each layer")
    parser.add_argument(
        "--beta", type = float,
        required = True,
        help = "beta value for generalized norm distribution")
    parser.add_argument(
        "--seed", type = float,
        required = True,
        help = "beta value for generalized norm distribution"
    )
    parser.add_argument(
        "--projected",type=str,
        required = False,
        choices=['huang_proj',''],
        default='',
        help = "use projected quantum kernel or not?")

    args = parser.parse_args()

    scaling_factor=args.scaling_factor
    int_time_scale=args.int_scaling_factor

    if args.projected != '':
        proj='_'+args.projected
    else:
        proj=args.projected

    outpath = Path(args.outpath, f"Sparse_IQP_dim_{args.dataset_dim}{proj}_scales_{scaling_factor}_{int_time_scale}_density_{args.density}_beta_{args.beta}_seed_{args.seed}.p")

    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    seed=args.seed
    samples = get_gennorm_samples(args.beta,args.dataset_dim,1000,seed)

    mu=np.mean(samples,axis=0)
    sigma=np.sqrt(np.var(samples,axis=0))
    #normalize and standardize
    samples=(samples-mu)/sigma
    #rescale using IQP parameter
    samples *= scaling_factor 

    FeatureMap = IQP.Sparse_IQP(args.dataset_dim,density=args.density,data_map_func=self_product,int_time_scale=int_time_scale)

    if args.projected=='':
        qkern = get_quantum_kernel(FeatureMap,device='CPU',batch_size=50)
        qkern_matrix_train = qkern.evaluate(x_vec=samples)
        rdms=None

    else:
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method='statevector',batch_size=10)
        mq=[[i] for i in range(args.dataset_dim)]
        qkern.set_measured_qubits(mq)
        qkern_matrix_train,rdms = qkern.evaluate(x_vec=samples,return_rdms=True)

    t1 = time.time()
    K_train_time = t1-t0
    print(f"Done computing K_train in {K_train_time}")

    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'args' : args,
            'K_train_time' : K_train_time,
            'rdms' : rdms,
            'samples' : samples,
            'mu' : mu,
            'sigma' : sigma,
    }

    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")