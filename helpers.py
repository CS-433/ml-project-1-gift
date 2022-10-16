import numpy as np
import matplotlib.pyplot as plt
import doctest
import io
import sys



def load_train_dataset():
    """Load data and convert it to the metric system."""
    path_dataset = "train.csv"
    
    data = np.genfromtxt(
            path_dataset, delimiter=",", skip_header=1)

    first_col = np.genfromtxt(
                path_dataset, delimiter=",", skip_header=1, usecols=[0])
        
    second_col = np.genfromtxt(
                path_dataset, delimiter=",", skip_header=1, usecols=[1],
                converters={1: lambda x: -1 if b"b" in x else 1})
    
    data[:, 0] = first_col
    data[:, 1] = second_col
    
    return data

#%%
def load_test_dataset():
    """Load data and convert it to the metric system."""
    path_dataset = "test.csv"
    
    data = np.genfromtxt(
            path_dataset, delimiter=",", skip_header=1)

    first_col = np.genfromtxt(
                path_dataset, delimiter=",", skip_header=1, usecols=[0])
        
    second_col = np.genfromtxt(
                path_dataset, delimiter=",", skip_header=1, usecols=[1],
                converters={1: lambda x: 0 if b"?" in x else 1})
    
    data[:, 0] = first_col
    data[:, 1] = second_col
    
    return data

#%%
def standardize_train(x):
    """Standardize the original data set."""
    ret = x.copy()
    mean = np.mean(ret[:,2:])
    std = np.std(ret[:,2:])
    ret[:,2:] = (ret[:,2:] - mean)/std
    return ret, mean, std
#%%
def standardize_test(x, mean_train, std_train):
    """Standardize the original data set."""
    ret = x.copy()
    ret[:,2:] = (ret[:,2:] - mean_train)/std_train
    
    return ret

#%% Count -999 values for each feature
def count_999_for_feature(dataset):
    
    ret = np.array([np.count_nonzero(dataset[:,i]==-999) for i in range(len(dataset[0,:]))])
    
    return ret

#%% Delete the rows with at least 1 -999 value among the columns
def delete_999_rows(dataset):
    
    check = []
    selec_rows = []

    for i in range(len(dataset)):
    
        check = (dataset[i] == -999)
        selec_rows.append(np.sum(check)== 0)
        
    ret = dataset[selec_rows]
        
    return ret

#%% Substitute -999 values with val
def subsitute_999_with_val(dataset, val):
    
    ret = dataset.copy()
    ret[ret==-999] = val
        
    return ret

#%% Substitute -999 values with the mean of the corresponding column
def subsitute_999_with_mean(dataset):
    
    ret = dataset.copy()

    for j in range(len(dataset[:,0])):
    
        indexes_no999 = (dataset[:,j] != -999)
        indexes_999 = (dataset[:,j] == -999)
        curr_col_wo999 = ret[indexes_no999,j]
        ret[indexes_999,j] = np.mean(curr_col_wo999)
        
    return ret



