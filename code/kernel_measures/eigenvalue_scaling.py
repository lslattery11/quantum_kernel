import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from quantum_kernel.code.visualization_utils import filter_df
import scipy
from scipy.stats.distributions import  t
from scipy.interpolate import interp1d

#ugly maybe rewrite if time allows
def get_eigenvalue_scaling(df: pd.DataFrame,gamma,lambdas,key='scaling_factor'):
    """
    Take data frame with max eigenvalue of kernel column,lambda column and dataset dim column.
    Create interpolation between max eigenvalue vs. N for each value of lambda. Find N returning gamma using 
    the interpolation. return List[(N,lambda)] for gamma.
    """
    assert((gamma > 0) & (gamma <= 1.0)),'gamma must be between 0 and 1.'
    fitted_points=[]
    for lam in lambdas:
        lam_df=df[(df[key]==lam)]

        x=lam_df.dataset_dim
        y=lam_df.apply(lambda x: max(x.kernel_eigenvalues),axis=1)
        points=np.array(list(zip(x.values,y.values)))
    
        new_x=list(set(points[:,0]))
        new_y=[np.mean([point[1] for point in points if point[0]==x]) for x in new_x]

        func=interp1d(new_x,new_y,kind='linear',bounds_error=False)

        try:
            func2 = lambda x: func(x)-gamma
            ypred=scipy.optimize.root_scalar(func2,x0=(max(new_x)+min(new_x))/2,bracket=[min(new_x),max(new_x)],method='brentq',xtol=0.1,maxiter=10**4).root
            fitted_points.append((ypred,lam))
        except:
            continue

    return np.array(fitted_points)



def compute_dataframe_kernel_eigenvalues(df: pd.DataFrame,filter: dict = {}, k: int = 1):
    """
    compute the kth largest kernel eigenvalues for a dataframe after filter. return filtered dataframe with new column 'kernel_eigenvalues'.
    """
    filtered_df=filter_df(df,filter)
    #-1* matrix so we get the 'kth lowest eigenvalues' then take absolute value to make positive.
    filtered_df['kernel_eigenvalues']=filtered_df.apply(lambda row: np.abs(scipy.linalg.eigh(-1*row.qkern_matrix_train,eigvals_only=True,subset_by_index=(0,k-1)))/row.qkern_matrix_train.shape[0], axis=1)
    return filtered_df