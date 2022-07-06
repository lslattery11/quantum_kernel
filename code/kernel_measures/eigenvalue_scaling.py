import numpy as np
import pandas as pd

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