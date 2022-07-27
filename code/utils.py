# assorted utils for QKernel experiments modified from https://github.com/rsln-s/Importance-of-Kernel-Bandwidth-in-Quantum-Machine-Learning/
import os
import gzip
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from functools import reduce
from scipy import stats

from quantum_kernel.code.quantum_kernels.projected_kernels import RDM1ProjectedKernel

utils_folder = Path(__file__).parent

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`
    Train: kind='train'
    Test: kind='t10k'
    """
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def get_fashion_mnist_dataset(dataset_dim, n_train, n_test):
    """ dataset from https://github.com/zalandoresearch/fashion-mnist
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    path = Path(utils_folder, '../data/fashion_mnist/')
    x_train, y_train = load_mnist(path, kind='train')
    x_test, y_test = load_mnist(path, kind='t10k')
    def filter_03(x, y):
        keep = (y == 0) | (y == 3)
        x, y = x[keep], y[keep]
        y = y == 0
        return x,y
    
    x_train, y_train = filter_03(x_train, y_train)
    x_test, y_test = filter_03(x_test, y_test)

    # normalize
    x_train, x_test = x_train/255.0, x_test/255.0
    feature_mean = np.mean(x_train,axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean

    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    x_train, x_test = x_train[:n_train], x_test[:n_test]
    y_train, y_test = y_train[:n_train], y_test[:n_test]
    return x_train, x_test, y_train, y_test


def get_kuzushiji_mnist_dataset(dataset_dim, n_train, n_test):
    """ dataset from https://github.com/rois-codh/kmnist
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    x_train = np.load(Path(utils_folder, '../data/kmnist/kmnist-train-imgs.npz'))['arr_0'].reshape(60000,784)
    y_train = np.load(Path(utils_folder, '../data/kmnist/kmnist-train-labels.npz'))['arr_0']
    x_test = np.load(Path(utils_folder, '../data/kmnist/kmnist-test-imgs.npz'))['arr_0'].reshape(10000,784)
    y_test = np.load(Path(utils_folder, '../data/kmnist/kmnist-test-labels.npz'))['arr_0']
    
    def filter_14(x, y):
        keep = (y == 1) | (y == 4)
        x, y = x[keep], y[keep]
        y = y == 1
        return x,y
    
    x_train, y_train = filter_14(x_train, y_train)
    x_test, y_test = filter_14(x_test, y_test)

    # normalize
    x_train, x_test = x_train/255.0, x_test/255.0
    feature_mean = np.mean(x_train, axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean

    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    x_train, x_test = x_train[:n_train], x_test[:n_test]
    y_train, y_test = y_train[:n_train], y_test[:n_test]
    return x_train, x_test, y_train, y_test


def get_plasticc_dataset(dataset_dim, n_train, n_test):
    """ dataset from https://arxiv.org/abs/2101.09581
    Normalized, downscaled to dataset_dim and truncated to n_train, n_test
    """
    data = np.load(open(Path(utils_folder, '../data/plasticc_data/SN_67floats_preprocessed.npy'), 'rb'))

    X = data[:,:67]
    Y = data[:,67]
    
    x_train_normalized, x_test_normalized, y_train, y_test = train_test_split(X, Y, train_size=n_train, test_size=n_test, random_state=42, stratify=Y)
    scikit_pca = PCA(n_components=dataset_dim)
    x_train = scikit_pca.fit_transform(x_train_normalized)
    x_test = scikit_pca.transform(x_test_normalized)
    return x_train, x_test, y_train, y_test


def get_dataset(name, dataset_dim, n_train, n_test):
    if name == 'fashion-mnist':
        return get_fashion_mnist_dataset(dataset_dim, n_train, n_test)
    elif name == 'kmnist':
        return get_kuzushiji_mnist_dataset(dataset_dim, n_train, n_test)
    elif name == 'plasticc':
        return get_plasticc_dataset(dataset_dim, n_train, n_test)
    else:
        raise ValueError(f"Unknown dataset: {name}")

def get_gennorm_samples(beta,dim,num_samples,seed=0):
    np.random.seed(seed)
    samples=np.zeros((num_samples,dim))
    for i in range(num_samples):
        sample=stats.gennorm(beta).rvs(dim)
        samples[i]=sample
    return samples

def get_quantum_kernel(FeatureMap, simulation_method='statevector', shots=1, batch_size=500,device='CPU',MPI=False):
    """Builds Qiskit QuantumKernel object 
    with parameters passed directly to HamiltonianEvolutionFeatureMap
    """
    from qiskit.providers.aer import AerSimulator
    from qiskit.utils import QuantumInstance
    from qiskit_machine_learning.kernels import QuantumKernel
    if simulation_method == 'statevector' and shots != 1:
        raise ValueError(f'With simulation method {simulation_method} no shots are allowed')
    if MPI==False:
        quantum_instance_sv = QuantumInstance(AerSimulator(method=simulation_method, shots=shots,device=device))
    else:
        quantum_instance_sv = QuantumInstance(AerSimulator(method=simulation_method, shots=shots,device=device,blocking_enable=True, blocking_qubits=20))
    return QuantumKernel(feature_map=FeatureMap, quantum_instance=quantum_instance_sv, batch_size=batch_size)

def get_projected_quantum_kernel(FeatureMap, simulation_method='statevector', shots=1, batch_size=500,gamma=1,device='CPU',MPI=False):
    """Builds my Qiskit projected QuantumKernel object
    with parameters passed directly to HamiltonianEvolutionFeatureMap
    """
    from qiskit.providers.aer import AerSimulator
    from qiskit.utils import QuantumInstance
    if simulation_method == 'statevector' and shots != 1:
        raise ValueError(f'With simulation method {simulation_method} no shots are allowed')
    if MPI==False:
        quantum_instance_sv = QuantumInstance(AerSimulator(method=simulation_method, shots=shots,device=device))
    else:
        quantum_instance_sv = QuantumInstance(AerSimulator(method=simulation_method, shots=shots,device=device,blocking_enable=True, blocking_qubits=23))
    return RDM1ProjectedKernel(feature_map=FeatureMap, quantum_instance=quantum_instance_sv, batch_size=batch_size,gamma=gamma)


def precomputed_kernel_GridSearchCV(K, y, Cs, gs, n_splits=5, test_size=0.2, random_state=42):
    """A version of grid search CV, 
    but adapted for SVM with a precomputed kernel
    K (np.ndarray) : precomputed kernel
    y (np.array) : labels
    Cs (iterable) : list of values of C to try
    gs (iterable) : list of values of gamma to try (if projected quantum kernel) else [0]
    return: optimal value of C
    """
    from sklearn.model_selection import ShuffleSplit

    n = K.shape[0]
    assert len(K.shape) == 2
    assert K.shape[1] == n
    assert len(y) == n
    
    best_score = float('-inf')
    best_C = None
    best_g = None

    indices = np.arange(n)
    
    for g in gs:
        for C in Cs:
            # for each value of parameter, do K-fold
            # The performance measure reported by k-fold cross-validation 
            # is the average of the values computed in the loop
            scores = []
            ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
            for train_index, test_index in ss.split(indices):
                K_train = K[np.ix_(train_index,train_index)]
                K_test = K[np.ix_(test_index, train_index)]
                if g != 0:
                    K_train = np.exp(-g*K_train)
                    K_test = np.exp(-g*K_test)
                y_train = y[train_index]
                y_test = y[test_index]
                svc = SVC(kernel='precomputed', C=C)
                svc.fit(K_train, y_train)
                scores.append(svc.score(K_test, y_test))
            if np.mean(scores) > best_score:
                best_score = np.mean(scores)
                best_C = C
                best_g = g
    return best_C,best_g

# Results DataFrame management routines
def get_additional_fields(row, datasets,projected):
    """
    row (one row of pd.DataFrame)
    datasets (dict): maps from dataset_dim to preloaded (x_train, x_test, y_train, y_test)
    """
    x_train, x_test, y_train, y_test = datasets[row['dataset_dim']]
    assert(row['qkern_matrix_train'].shape == (len(x_train),len(x_train)))
    C_range = [0.006, 0.015, 0.03, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256, 512, 1024]

    if projected==True:
        gamma_range=np.linspace(0,100,num=40)
    else:
        gamma_range=[0]

    C_opt,gamma_opt = precomputed_kernel_GridSearchCV(row['qkern_matrix_train'], y_train, C_range,gamma_range)
    
    if projected==True:
        kernel_train=np.exp(-gamma_opt*row['qkern_matrix_train'])
        kernel_test=np.exp(-gamma_opt*row['qkern_matrix_test'])
    else:
        kernel_train=row['qkern_matrix_train']
        kernel_test=row['qkern_matrix_test']

    svc = SVC(kernel='precomputed', C=C_opt)
    svc.fit(kernel_train, y_train)
    y_pred_train = svc.predict(kernel_train)
    train_score = balanced_accuracy_score(y_train,y_pred_train)

    y_pred_test = svc.predict(kernel_test)
    test_score = balanced_accuracy_score(y_test,y_pred_test)

    n_support = svc.n_support_
    n_support_ave = np.mean(n_support)
    norm_K_id = np.linalg.norm(kernel_train - np.eye(kernel_train.shape[0]))
    return test_score, train_score, n_support, n_support_ave, C_opt, gamma_opt, norm_K_id

def compute_additional_fields(df, dataset_name,kernel_name=None,projected=False):
    datasets = {}
    for dataset_dim in set(df['dataset_dim']):
        datasets[dataset_dim] = get_dataset(dataset_name, dataset_dim, 800, 200)
    
    temp_name=df.progress_apply(lambda row: get_additional_fields(row, datasets,projected),axis=1,result_type="expand",)
    df[[
        'test_score',
        'train_score', 
        'n_support',
        'n_support_ave',
        'C',
        'proj_gamma',
        'norm(qkern_matrix_train - identity)',
    ]] = temp_name

    df['kernel_name']=kernel_name
    return df

def self_product(x: np.ndarray) -> float:
    """
    copied from
    https://qiskit.org/documentation/_modules/qiskit/circuit/library/data_preparation/pauli_feature_map.html#PauliFeatureMap
    Define a function map from R^n to R.
    Args:
        x: data
    Returns:
        float: the mapped value
    """
    coeff = x[0] if len(x) == 1 else reduce(lambda m, n: m * n, x)
    return coeff