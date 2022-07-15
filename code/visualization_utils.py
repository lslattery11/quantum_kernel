import pickle
import pandas as pd
from pathlib import Path
import numpy as np
import copy
import sys
import os
import seaborn as sns
import scipy
import matplotlib.pyplot as plt


from tqdm import tqdm
tqdm.pandas()

from quantum_kernel.code.utils import compute_additional_fields

def aggregate_pickles(all_pickles_paths, dataset_name,kernel_name,projected=False):
    all_res = []
    
    for fname in all_pickles_paths:
        try:
            res = pickle.load(open(fname,'rb'))
        except (AttributeError, EOFError, TypeError) as e:
            print(e)
            print(fname)
            continue
        res.update(vars(res['args']))
        all_res.append(res)

    df_all = pd.DataFrame(all_res, columns=all_res[0].keys())
    df_all = compute_additional_fields(df_all, dataset_name=dataset_name,kernel_name=kernel_name,projected=projected)
    return df_all   

#ugly should rewrite eventually
def aggregate_folder(folder,dataset_name,kernel_name,projected=False):
    dfs={}

    label = Path(folder).stem
    if "Sparse_IQP" in folder:
        prefix = "Sparse_IQP"
    else:
        prefix = "dim"
    all_pickles_paths = list(Path(folder).glob(f"{prefix}*.p"))
    npickles = len(all_pickles_paths)
    # check if the data in the pickles has been aggregated before
    # if not, compute an aggregated pickle with all the extra pickles
    must_reaggregate = True
    path_aggregated = Path(folder, "aggregated.p")
    if path_aggregated.exists():
        aggregated_df = pickle.load(open(path_aggregated, "rb"))
        if len(aggregated_df) == npickles:
            must_reaggregate = False
            print(f"For {folder}, using aggregated pickle from {path_aggregated}")
            dfs[label] = copy.deepcopy(aggregated_df)
    if must_reaggregate:
        aggregated_df = aggregate_pickles(all_pickles_paths, dataset_name,kernel_name,projected=projected)
        dfs[label] = copy.deepcopy(aggregated_df)
        print(f"For {folder}, saving aggregated pickle in {path_aggregated}")
        pickle.dump(aggregated_df, open(path_aggregated, "wb"))
    dfs[label] = dfs[label][dfs[label]['dataset_dim'] <= 22] 

    if "Sparse_IQP" in folder:
        dfs[label]['Number of qubits'] = dfs[label]['dataset_dim']
    else:
        dfs[label]['Number of qubits'] = dfs[label]['dataset_dim'] + 1
    return dfs

#
def aggregate_shapes(folder,prefix,return_dataframe=True,cols_to_drop=None):
    all_pickles_paths = list(Path(folder).glob(f"{prefix}*.p"))
    all_res = []
    df_all = pd.DataFrame()
    for fname in all_pickles_paths:
        try:
            res = pickle.load(open(fname,'rb'))
        except (AttributeError, EOFError, TypeError) as e:
            print(e)
            print(fname)
            continue
        res.update(vars(res['args']))

        all_res.append(res)
        if len(all_res) >= 500:
            df_all=update_df(df_all,all_res,cols_to_drop)
            all_res=[]

    df_all=update_df(df_all,all_res,cols_to_drop)
    
    if return_dataframe==True:
        return df_all

#return df with filter applied.
def filter_df(
    df: pd.DataFrame,
    df_filter_dict: dict,
    ):
    filtered_df=df.loc[(df[list(df_filter_dict)] == pd.Series(df_filter_dict)).all(axis=1)]
    return filtered_df

def update_df(df,res_list,cols_to_drop,k=1):
    if df.columns.size==0:
        df=pd.DataFrame(res_list, columns=res_list[0].keys())
        df['kernel_eigenvalues']=df.apply(lambda row: np.abs(scipy.linalg.eigh(-1*row.qkern_matrix_train,eigvals_only=True,subset_by_index=(0,k-1)))/row.qkern_matrix_train.shape[0], axis=1)

    else:
        temp_df=pd.DataFrame(res_list, columns=res_list[0].keys())
        temp_df['kernel_eigenvalues']=temp_df.apply(lambda row: np.abs(scipy.linalg.eigh(-1*row.qkern_matrix_train,eigvals_only=True,subset_by_index=(0,k-1)))/row.qkern_matrix_train.shape[0], axis=1)

        df=df.append(temp_df,ignore_index=True)
    if cols_to_drop is not None:
        df=df[df.columns[~df.columns.isin(cols_to_drop)]]
    return df