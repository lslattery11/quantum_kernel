import numpy as np
import scipy
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
#Relevant kernel measures from 'Power of data in quantum machine learning' https://arxiv.org/abs/2011.01938.

#trained model complexity, smaller value implies better generalization to new data. Eq. 4 from paper. Not implemented yet.
def s_complexity(K):
    raise NotImplementedError('have not finished function')
    return

#geometric difference (for generalization error) of kernel matrices. Eq. 5 from paper (See appendix F for inclusion of regularization parameter lambda.)
#Note they reorder K1 and K2 in the appendix which is NOT the convention we follow here.
#THE ORDER OF THE INPUTS MATTERS. We use gcq(K1||K2) with the (main text notation)
def geometric_difference(K1,K2,lam=0,tol=1e-3,**kwargs):
    assert(K1.shape==K2.shape),'K1 and K2 must be the same shape'
    assert(np.allclose(np.trace(K1),len(K1),atol=tol)),'K1 must be normalized Tr(K1)==N'
    assert(np.allclose(np.trace(K2),len(K2),atol=tol)),'K2 must be normalized Tr(K2)==N'
    #Could add an assertion that Ks are positive semidefinite and symmetric. Probably unecessary as they will be by construction (maybe do later).
    K1root=scipy.linalg.sqrtm(K1)
    K2root=scipy.linalg.sqrtm(K2)
    K1inv2=np.linalg.matrix_power(K1+lam*np.eye(K1.shape[0]),-2)
    #multiply matrices
    multiplied=np.linalg.multi_dot([K2root,K1root,K1inv2,K1root,K2root])
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
    K2inv2=np.linalg.matrix_power(K2+lam*
    np.eye(K2.shape[0]),-2)
    #multiply matrices
    multiplied=np.linalg.multi_dot([K1root,K2inv2,K1root])
    #compute spectral norm, take root, multiply by lam
    gd=lam*np.linalg.norm(multiplied,ord=np.inf)**(1/2)
    return gd

#alternative metric instead of geometric difference which they call 'geometric distance'. Eq. 1 in https://arxiv.org/abs/1806.01428
#This is a symmetric metric (unlike geometric difference which is a 'divergence') and has nice properties like
#it is invariant under transformations to matrices A,B like A-> XAXdag where X is an invertible matrix and A -> Ainv
# A <-> B.
def geometric_distance(K1,K2,tol=1e-3,**kwargs):
    assert(K1.shape==K2.shape),'K1 and K2 must be the same shape'
    assert(np.allclose(np.trace(K1),len(K1),atol=tol)),'K1 must be normalized Tr(K1)==N'
    assert(np.allclose(np.trace(K2),len(K2),atol=tol)),'K2 must be normalized Tr(K2)==N'
    K1inv=np.linalg.inv(K1)
    M=K1inv @ K2
    w=np.linalg.eigvals(M)
    gd=np.sqrt(np.sum(np.log(w)**2))
    return np.real(gd)

#distance between subspace spanned by the k largest eigenvectors of two kernels.
#the eigenvectors are in Real^n so the distance between the A=span(k eigenvectors1) and B=span(k eigenvectors2)
#is given by the Grassmann distance dk(A,B)
#https://web.ma.utexas.edu/users/vandyke/notes/deep_learning_presentation/presentation.pdf
def grassmann_distance(K1,K2,k=10,tol=1e-3):
    assert(K1.shape==K2.shape),'K1 and K2 must be the same shape'
    assert(np.allclose(np.trace(K1),len(K1),atol=tol)),'K1 must be normalized Tr(K1)==N'
    assert(np.allclose(np.trace(K2),len(K2),atol=tol)),'K2 must be normalized Tr(K2)==N'
    #set default value of k to size of matrix (i.e. all of the eigenvectors).
    if k==None:
        k=K1.shape[0]
    _,v1 = np.linalg.eigh(K1)
    _,v2 = np.linalg.eigh(K2)
    #keep the largest eigenvectors
    v1=v1[-1:(-1+k):-1]
    v2=v2[-1:(-1+k):-1]
    #these are eigenvectors of a matrix so they are all orthogonromal o each other so we don't have to worry about
    #enforcing the condition of orthonormality. We want the the ordered pair of eigenvectors (v1_i,v2_i) such that
    #cos(theta_i)=np.dot(v1_i,  v2_i) and theta_1 <= theta_2 ...etc.
    #compute all dot product pairs.
    theta_matrix=np.arccos(cosine_similarity(v1,v2))
    #Note that if v1_i and v2_i form a min pair then v1_i & v2_j will not form the second largest pair etc. I'm not going to
    #do this. Maybe later. Should check math.
    thetas=np.zeros(k,)
    for i in range(k):
        theta=np.min(theta_matrix)
        row,col=np.where(theta_matrix == theta)
        theta_matrix=np.delete(theta_matrix,row,0)
        theta_matrix=np.delete(theta_matrix,col,1)
        thetas[i]=theta
    grass_d=np.sum(np.power(thetas,2))**(1/2)
    return grass_d

#takes as input a pd.Dataframe of kernels and one of the kernel model hyperparameters.
#for each kernel, compute the k largest eigenvalue,eigenvector pairs. returns 
#the sorted array of hyperparaemters and a
# a vector V with elements V[i] = (eigenvalues,eigenvectors) for the ith hyperparameter.
def compute_eigenvector_matrix(df: pd.DataFrame,k=5):
    V=np.zeros((len(df),),dtype=object)
    assert(len(df.columns)==2),'number of keys must be 2'
    h=np.setdiff1d(df.columns.values,np.array(['qkern_matrix_train']))[0]
    #sort DataFrame by hyper parameter
    sorted_df=df.sort_values(h)
    x=sorted_df[h]
    #compute matrix
    for i,a in enumerate(x):
        K=sorted_df[sorted_df[h] == a]['qkern_matrix_train'].values[0]
        w,v = np.linalg.eigh(K)
        V[i]=(w[0:k],v[0:k])
    return x, V

#takes as input two pd.Dataframes of kernels and one of the kernel model hyperparameters.
#compute the matrix of the relevant metric i.e. Mij=gd(k1[i]||k2[j]) for geometric difference
def compute_metric_matrix(df1: pd.DataFrame,df2: pd.DataFrame,metric,k=10):
    metric_dict={'geometric_difference':geometric_difference,'geometric_distance':geometric_distance,'grassmann_distance':grassmann_distance}
    M=np.zeros((len(df2),len(df1)))
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
            M[j][i]=f(K1,K2,k=k)
    #return sorted hyper parameters and metric matrix.
    return x, y, M

def compute_kernel_target_alignment(K):
    raise NotImplementedError('have not finished function')
    return

#Eq. 6 in https://arxiv.org/pdf/2006.13198.pdf
def compute_task_model_alignment(K,target):
    raise NotImplementedError('have not finished function')
    return

def compute_cumulative_power_distribution(K):
    w,_=np.linalg.eig(K)
    return np.cumsum(w**2)/np.sum(w**2)

#return tdm_average, a np.array with shape ((rdms.shape[1],)) where the elements are
# tdm_average[i]=<np.trace(rdm[:,i]-mixed_state)>
#better way to do this with np.array without for loops but doesn't matter for our purposes.
def tdm_average(rdms,size=2):
    mixed_state=1/size*np.eye(size)
    tdm_average=np.zeros((rdms.shape[1],))
    for i in range(len(tdm_average)):
        traces=[]
        for rdm in rdms[:,i]:
            traces.append(np.trace(rdm-mixed_state))
        tdm_average[i]=np.average(traces)
    return tdm_average

#return tdm_of_average, a np.array with shape ((rdms.shape[1],)) where the elements are
# tdm_of_average[i]=np.trace(<rdm[:,i]>-mixed_state)
def tdm_of_average(rdms,size=2):
    mixed_state=1/size*np.eye(size)
    tdm_of_average=np.zeros((rdms.shape[1],))
    for i in range(len(tdm_of_average)):
        rdm_avg=1/rdms.shape[0]*np.sum(rdms[:,i],axis=0)
        tdm_of_average[i]=np.trace(rdm_avg-mixed_state)
    return tdm_of_average

#return purity_average, a np.array with shape ((rdms.shape[1],)) where the elements are
# purity_average[i]=<np.trace(rdm[:,i]^2)>
#better way to do this with np.array without for loops but doesn't matter for our purposes.
def purity_average(rdms):
    purity_average=np.zeros((rdms.shape[1],))
    for i in range(len(purity_average)):
        traces=[]
        for rdm in rdms[:,i]:
            traces.append(np.trace(rdm @ rdm))
        purity_average[i]=np.real(np.average(traces))
    return purity_average

#return purity_of_average, a np.array with shape ((rdms.shape[1],)) where the elements are
# purity_of_average[i]=np.trace(<rdm[:,i]>^2)
#better way to do this with np.array without for loops but doesn't matter for our purposes.
def purity_of_average(rdms):
    purity_of_average=np.zeros((rdms.shape[1],))
    for i in range(len(purity_of_average)):
        rdm_avg=1/rdms.shape[0]*np.sum(rdms[:,i],axis=0)
        purity_of_average[i]=np.real(np.trace(rdm_avg @ rdm_avg))
    return purity_of_average

#approximate dimension d given by equation F12 in Power of Data paper. Inefficient but whatever.
def approximate_matrix_dimesnion(K,N):
    d=0
    w,_ = np.linalg.eigh(K)
    w=w[::-1]
    for k in range(1,N+1):
        for l in range(k,N+1):
            d+=1/(N-k)*w[l-1]
    return d

