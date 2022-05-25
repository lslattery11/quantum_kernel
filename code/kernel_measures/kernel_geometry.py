import numpy as np
import scipy
import pandas as pd
#Relevant kernel measures from 'Power of data in quantum machine learning' https://arxiv.org/abs/2011.01938.

#trained model complexity, smaller value implies better generalization to new data. Eq. 4 from paper. Not implemented yet.
def s_complexity(K):
    return

#geometric difference (for generalization error) of kernel matrices. Eq. 5 from paper (See appendix F for inclusion of regularization parameter lambda.)
#Note they reorder K1 and K2 in the appendix which is NOT the convention we follow here.
#larger lam leads to smaller generalization geometric difference.
#THE ORDER OF THE INPUTS MATTERS. TO prepare  notation prepares gcq(K2||K1)(main text notation)
def geometric_difference(K2,K1,lam=0,tol=1e-3):
    assert(K1.shape==K2.shape),'K1 and K2 must be the same shape'
    assert(np.allclose(np.trace(K1),len(K1),atol=tol)),'K1 must be normalized Tr(K1)==N'
    assert(np.allclose(np.trace(K2),len(K2),atol=tol)),'K2 must be normalized Tr(K2)==N'
    #Could add an assertion that Ks are positive semidefinite and symmetric. Probably unecessary as they will be by construction (maybe do later).
    K1root=scipy.linalg.sqrtm(K1)
    K2root=scipy.linalg.sqrtm(K2)
    K2inv2=np.linalg.matrix_power(K2+lam*np.eye(K2.shape[0]),-2)
    #multiply matrices
    multiplied=np.linalg.multi_dot([K1root,K2root,K2inv2,K2root,K1root])
    #compute spectral norm and take root.
    gd=np.linalg.norm(multiplied,ord=np.inf)**(1/2)
    return gd

#geometric difference (for training error) of kernel matrices. (In text after Eq. F20)
#larger lam leads to larger training geometric differnce.
def training_geometric_difference(K1,K2,lam=0,tol=1e-3):
    assert(K1.shape==K2.shape),'K1 and K2 must be the same shape'
    assert(np.allclose(np.trace(K1),len(K1),atol=tol)),'K1 must be normalized Tr(K1)==N'
    assert(np.allclose(np.trace(K2),len(K2),atol=tol)),'K2 must be normalized Tr(K2)==N'
    #Could add an assertion that Ks are positive semidefinite and symmetric. Probably unecessary as they will be by construction (maybe do later).
    K1root=scipy.linalg.sqrtm(K1)
    K2inv2=np.linalg.matrix_power(K2+lam*np.eye(K2.shape[0]),-2)
    #multiply matrices
    multiplied=np.linalg.multi_dot([K1root,K2inv2,K1root])
    #compute spectral norm, take root, multiply by lam
    gd=lam*np.linalg.norm(multiplied,ord=np.inf)**(1/2)
    return gd

#takes as input two pd.Dataframes of kernels and 1 of their hyperparameters
# and compute the matrix of their geometric difference Mij=gd(k1[i]||k2[j])
def test(kseries1: pd.Series,kseries2: pd.Series,lam=0):
    Mij=np.zeros((len(kseries1),len(kseries2)))
    
    return
