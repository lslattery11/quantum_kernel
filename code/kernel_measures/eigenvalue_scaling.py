import numpy as np
import pandas as pd
import scipy
from quantum_kernel.code.visualization_utils import filter_df

def get_eigenvalue_scaling(df: pd.DataFrame,gamma,lambdas):
    """
    Take data frame with max eigenvalue of kernel column,lambda column and dataset dim column.
    Create interpolation between max eigenvalue vs. N for each value of lambda. Find N returning gamma using 
    the interpolation. return List[(N,lambda)] for gamma.
    """
    assert((gamma > 0) & (gamma <= 1.0)),'gamma must be between 0 and 1.'
    points=[]
    for lam in lambdas:
        lam_df=df[(df['scaling_factor']==lam)]
        #note xp and yp are different than lambda vs. N normally plotted because we want to interpolate to find the x intercept.
        yp=lam_df.dataset_dim
        xp=lam_df.apply(lambda x: max(x.kernel_eigenvalues),axis=1)
        #sort for interpolation
        xp,yp=zip(*sorted(zip(xp.values, yp.values), key=lambda x: x[0]))

        n=np.interp(gamma,xp,yp)
        
        points.append((n,lam))
    return np.array(points)

def compute_dataframe_kernel_eigenvalues(df: pd.DataFrame,filter: dict = {}, k: int = 1):
    """
    compute the kth largest kernel eigenvalues for a dataframe after filter. return filtered dataframe with new column 'kernel_eigenvalues'.
    """
    filtered_df=filter_df(df,filter)
    #-1* matrix so we get the 'kth lowest eigenvalues' then take absolute value to make positive.
    filtered_df['kernel_eigenvalues']=filtered_df.apply(lambda row: np.abs(scipy.linalg.eigh(-1*row.qkern_matrix_train,eigvals_only=True,subset_by_index=(0,k-1)))/row.qkern_matrix_train.shape[0], axis=1)
    return filtered_df