import numpy as np
import matplotlib.pyplot as plt
import doctest
import io
import sys
import csv
from itertools import zip_longest



#%% load the train set
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


#%% load the test set
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


#%% split the whole dataset into y (output) and tx (design matrix)
def split_into_y_tx(dataset):
    
    y = dataset[:,1].copy()
    tx = dataset[:, 2:].copy()
    
    return y, tx

#%% standardize the train set
def standardize_train(x):
    """Standardize the original data set."""
    ret = x.copy()
    mean = np.mean(ret[:,2:])
    std = np.std(ret[:,2:])
    ret[:,2:] = (ret[:,2:] - mean)/std
    return ret, mean, std


#%% standardize the test set
def standardize_test(x, mean_train, std_train):
    """Standardize the original data set."""
    ret = x.copy()
    ret[:,2:] = (ret[:,2:] - mean_train)/std_train
    
    return ret


#%% Count -999 values for each feature
def count_nan_for_feature(dataset):
    
    n_cols = dataset[0,:].size
    ind_col = np.arange(n_cols)
    ret = np.array([np.count_nonzero(dataset[:,j]==-999) for j in ind_col])
    
    return ret


#%% delete the features whose -999 values are more than 177000
def delete_feature_with_177k(dataset):
    
    n_features = dataset[0,:].size -2
    cnt = count_nan_for_feature(dataset)
    feat_to_keep = (cnt<177000)
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

#%%
def delete_deriv(dataset):
    
    feat_to_keep = np.array([0, 1, 15, 16, 17, 18, 19, 20, 21, 22,
                             23, 24, 25, 26, 27, 28, 29, 30, 31])
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

#%% Delete the rows with at least 1 -999 value among the columns
def delete_999_rows(dataset):
    
    check = []
    selec_rows = []

    for i in range(len(dataset)):
    
        check = (dataset[i] == -999)
        selec_rows.append(np.sum(check)== 0)
        
    ret = dataset[selec_rows].copy()
        
    return ret

#%% Substitute -999 values with val
def subsitute_nan_with_val(dataset, val):
    
    ret = dataset.copy()
    ret[ret==-999] = val
        
    return ret

#%% Substitute -999 values with the mean of the corresponding column
def subsitute_nan_with_mean(dataset):
    
    ret = dataset.copy()

    for j in range(len(dataset[:,0])):
    
        indexes_no999 = (dataset[:,j] != -999)
        indexes_999 = (dataset[:,j] == -999)
        curr_col_wo999 = ret[indexes_no999,j]
        ret[indexes_999,j] = np.mean(curr_col_wo999)
        
    return ret


#%% generating a prediction from the weights and the design matrix
def generate_prediction(tx, w):
    
    prediction = tx.dot(w)
    
    index_neg = (prediction < 0)
    index_pos = (prediction >= 0)
    
    prediction[index_neg] = -1
    prediction[index_pos] = 1

    return prediction


#%% generate the output predicion file
def generate_csv(prediction):
    
    Id = np.arange(350000,918238)
    Prediction = prediction

    data = [Id, Prediction]
    export_data = zip_longest(*data, fillvalue = '')
    with open('sample-submission.csv', 'w', encoding="ISO-8859-1", newline='') as file:
        write = csv.writer(file)
        write.writerow(("Id", "Prediction"))
        write.writerows(export_data)
        
    return



