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

#takes as input two pd.Dataframes of kernels and one of their hyperparameters
# and compute the matrix of the relevant metric i.e. Mij=gd(k1[i]||k2[j]) for geometric difference
def compute_metric_matrix(df1: pd.DataFrame,df2: pd.DataFrame,metric):
    metric_dict={'geometric_difference':geometric_difference}
    M=np.zeros((len(df1),len(df2)))
    assert(len(df1.columns)==2),'number of keys must be 2'
    assert(len(df2.columns)==2),'number of keys must be 2'
    #get hyperparameter names
    h1=np.setdiff1d(df1.columns.values,np.array(['qkern_matrix_train']))[0]
    h2=np.setdiff1d(df2.columns.values,np.array(['qkern_matrix_train']))[0]
    #sort DataFrame by hyper parameter
    sorted_df1=df1.sort_values(h1)
    sorted_df2=df2.sort_values(h2)
    x=sorted_df1[h1]
    y=sorted_df2[h2]
    #get relevant function for metric
    f=metric_dict[metric]
    #compute geometric difference matrix
    for i,a in enumerate(x):
        for j,b in enumerate(y):
            K1=sorted_df1[sorted_df1[h1] == a]['qkern_matrix_train'].values[0]
            K2=sorted_df2[sorted_df2[h2] == b]['qkern_matrix_train'].values[0]
            M[i][j]=f(K1,K2)
    #return sorted hyper parameters and metric matrix.
    return x, y, M
