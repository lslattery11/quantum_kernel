import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from quantum_kernel.code.visualization_utils import filter_df
import scipy
from scipy.stats.distributions import  t

#ugly maybe rewrite if time allows
def get_eigenvalue_scaling(df: pd.DataFrame,gamma,lambdas):
    """
    Take data frame with max eigenvalue of kernel column,lambda column and dataset dim column.
    Create interpolation between max eigenvalue vs. N for each value of lambda. Find N returning gamma using 
    the interpolation. return List[(N,lambda)] for gamma.
    """
    assert((gamma > 0) & (gamma <= 1.0)),'gamma must be between 0 and 1.'
    fitted_points=[]
    for lam in lambdas:
        lam_df=df[(df['scaling_factor']==lam)]
        #note xp and yp are different than lambda vs. N normally plotted because we want to interpolate to find the x intercept.
        xp=lam_df.dataset_dim
        xp=np.array(list(set(xp.values)))
        #sort for interpolation
        points=[]
        for x in xp:
            ys=lam_df[(lam_df['dataset_dim']==x)].apply(lambda x: max(x.kernel_eigenvalues),axis=1).values
            xs=np.array(len(ys)*[x])
            new_points=list(zip(xs,ys))
            points=points+new_points
        points=np.array(points) 
        #fit exponential decay to the points.
        x=points[:,0]
        y=points[:,1]

        #popt,pcov=scipy.optimize.curve_fit(lambda t,a,b: a+b*t,  x,  np.log(y))
        #popt,pcov=scipy.optimize.curve_fit(lambda t,a,b,c: np.log(a*np.exp(-b*t)+c),  x,  np.log(y),bounds=(0, [1.0, 5, 0.2]))
        popt,pcov=scipy.optimize.curve_fit(lambda t,a,b,c,d: np.log(a*np.exp(-b*t)+c+d*x),  x,  np.log(y),bounds=([0,0,0,-2], [1.0, 2.0, 0.5,2]))

        #solution np.log(y)=a+b*x -> y=exp(a)*exp(b*x)
        #now solve for x, np.log(gamma) = a+b*x -> x = (np.log(gamma)-a)/b
        #ypred=(np.log(gamma)-popt[0])/popt[1]
        ypred=-np.log((gamma-popt[2])/popt[0])/popt[1]
        #prediction interval t(1-a/2,number of data points - number of independent variables -1). 95% confidence interval.
        fitted_points.append((ypred,lam))

    return np.array(fitted_points)



def compute_dataframe_kernel_eigenvalues(df: pd.DataFrame,filter: dict = {}, k: int = 1):
    """
    compute the kth largest kernel eigenvalues for a dataframe after filter. return filtered dataframe with new column 'kernel_eigenvalues'.
    """
    filtered_df=filter_df(df,filter)
    #-1* matrix so we get the 'kth lowest eigenvalues' then take absolute value to make positive.
    filtered_df['kernel_eigenvalues']=filtered_df.apply(lambda row: np.abs(scipy.linalg.eigh(-1*row.qkern_matrix_train,eigvals_only=True,subset_by_index=(0,k-1)))/row.qkern_matrix_train.shape[0], axis=1)
    return filtered_df