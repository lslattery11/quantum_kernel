"""
Computes quantum kernel and dumps everything on disk

"""

import argparse
import sys
import time
import numpy as np
import pickle
from pathlib import Path
from sklearn.svm import SVR

from quantum_kernel.code.utils import get_dataset,get_quantum_kernel,get_projected_quantum_kernel,get_quantum_data
from quantum_kernel.code.visualization_utils import aggregate_quantum_data
from quantum_kernel.code.feature_maps import HeisenbergZZMultiScale

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outpath", type = Path,
        required = True,
        help = "folder to dump the result")
    parser.add_argument(
        "--datapath", type = Path,
        required = True,
        help = "folder to quantum data")
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
        "--target_lam0", type = float,
        required = True,
        help = "target data kinetic term scale")
    parser.add_argument(
        "--target_lam1", type = float,
        required = True,
        help = "target non-data interaction term scale")
    parser.add_argument(
        "--target_lam2", type = float,
        required = True,
        help = "taret data interaction term scale")
    parser.add_argument(
        "--noise", type = float,
        required = True,
        help = 'gaussian noise level on dataset.'
    )
    parser.add_argument(
        "--projected",type=str,
        required = False,
        choices=['huang_proj','single_proj',''],
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

    outpath = Path(args.outpath, f"dim_{args.dataset_dim}{proj}_ntrot_{args.n_trotter}_evo_t_{args.evo_time}_init_{args.init_state}_s_{args.init_state_seed}_ls_{args.lam0}{args.lam1}{args.lam2}_tls{args.target_lam0}{args.target_lam1}{args.target_lam2}_{args.simulation_method}.p")
    if outpath.exists():
        print(f"Found already computed at {outpath}, exiting")
        sys.exit()
    else:
        print(f"Using outpath: {outpath}")
    t0 = time.time()

    if args.dataset_dim > 30:
        raise ValueError(f"Dataset dimension {args.dataset_dim} too large; can support no more than 30 qubits")

    #get dataframe of quantum_data
    folder=args.datapath
    prefix='MultiScale'
    df=aggregate_quantum_data(folder,prefix)
    #get quantum data we want from dataframe
    filter={'seed':0,
        'regression':True,
        'operators': ['Z'],
        'rdm_qubits':[[0]],
        'dataset_dim':args.dataset_dim,
        'n_trotter':args.n_trotter,
        'evo_time':args.evo_time,
        'init_state':args.init_state,
        'init_state_seed':args.init_state_seed,
        'lam0':args.target_lam0,
        'lam1':args.target_lam1,
        'lam2':args.target_lam2,
        'n_points':1000,
        'noise': args.noise,
        }
    n_train=800
    n_test=200
    x_train, x_test, y_train, y_test = get_quantum_data(df,filter,n_train,n_test)
    
    FeatureMap= HeisenbergZZMultiScale.HZZMultiscaleFeatureMap(args.dataset_dim,args.n_trotter,args.evo_time,args.init_state,args.init_state_seed,args.lam0,args.lam1,args.lam2)
    scaling_factor=FeatureMap.scaling_factor
    x_train *= scaling_factor 
    x_test *= scaling_factor


    if args.projected=='':
        qkern = get_quantum_kernel(FeatureMap,device='CPU',simulation_method=args.simulation_method,shots=args.shots,batch_size=10)
        qkern_matrix_train = qkern.evaluate(x_vec=x_train)
        rdms=None
    elif args.projected=='huang_proj':
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method=args.simulation_method,shots=args.shots,batch_size=10)
        mq=[[i] for i in range(args.dataset_dim)]
        qkern.set_measured_qubits(mq)
        qkern_matrix_train,rdms = qkern.evaluate(x_vec=x_train,return_rdms=True)
    elif args.projected=='single_proj':
        qkern = get_projected_quantum_kernel(FeatureMap,device='CPU',simulation_method=args.simulation_method,shots=args.shots,batch_size=10)
        qkern_matrix_train,rdms = qkern.evaluate(x_vec=x_train,return_rdms=True)


    t1 = time.time()
    K_train_time = t1-t0
    print(f"Done computing K_train in {K_train_time}")

    qkern_matrix_test = qkern.evaluate(x_vec=x_test, y_vec=x_train)
    t2 = time.time()
    K_test_time = t2-t1
    print(f"Done computing K_test in {K_test_time}")

    res = {
            'qkern_matrix_train' : qkern_matrix_train,
            'qkern_matrix_test' : qkern_matrix_test,
            'args' : args,
            'K_train_time' : K_train_time,
            'K_test_time' : K_test_time,
            'rdms' : rdms,
            'filter': filter,
    }
    
    pickle.dump(res, open(outpath, 'wb'))
    print(f"Dumped the result to {outpath}")