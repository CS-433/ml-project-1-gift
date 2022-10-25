import numpy as np
import matplotlib.pyplot as plt
import doctest
import io
import sys
import csv
from itertools import zip_longest
from utilities import *
from implementations import *


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
    col24 = data[:, 24]
    data = np.delete(data, 24, 1)
    
    return data, col24
    
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
    col24 = data[:, 24]
    data = np.delete(data, 24, 1)
    
    return data, col24

def create_dummy(col24):
    cols = np.zeros((col24.size, 4))
    for i in range(len(col24)):
        cols[i][int(col24[i])] = 1
        
    return cols

def split_into_ids_y_tx(dataset):
    
    ids = dataset[:, 0].copy()
    y = dataset[:, 1].copy()
    tx = dataset[:, 2:].copy()
    
    return ids, y, tx

def standardize_train(x):
    """Standardize the original data set."""
    ret = x.copy()
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    ret[:,2:] = (ret[:,2:] - mean)/std
    return ret, mean, std

def standardize_test(x, mean_train, std_train):
    """Standardize the original data set."""
    ret = x.copy()
    ret[:,2:] = (ret[:,2:] - mean_train)/std_train
    
    return ret

def count_nan_for_feature(dataset):
    
    n_cols = dataset[0,:].size
    ind_col = np.arange(n_cols)
    ret = np.array([np.count_nonzero(dataset[:,j]==-999) for j in ind_col])
    
    return ret

def count_nan_for_elem(col_vector):
    
    ret = np.count_nonzero(col_vector==-999)
    
    return ret

def delete_feature_with_50pec(dataset):
    
    n = dataset[:, 0].size
    cnt = count_nan_for_feature(dataset)
    feat_to_keep = (cnt < 0.5*n)
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

def delete_deriv(dataset):
    
    feat_to_keep = np.array([0, 1, 15, 16, 17, 18, 19, 20, 21, 22,
                             23, 24, 25, 26, 27, 28, 29, 30])
    ret = dataset[:, feat_to_keep].copy()
    
    return ret

def delete_nan_rows(dataset):
    
    check = []
    selec_rows = []

    for i in range(len(dataset[:,0])):
    
        check = np.count_nonzero(dataset[i,:]==-999)
        
        if(check==0):
            selec_rows.append(i)
        
    ret = dataset[selec_rows].copy()
    selec_rows = np.array(selec_rows)
        
    return ret, selec_rows

def delete_nan_elem(col_vector):
    
    check = []
    selec_rows = []

    for i in range(len(col_vector)):
    
        check = np.count_nonzero(col_vector[i]==-999)
        
        if(check==0):
            selec_rows.append(i)
        
    ret = col_vector[selec_rows].copy()
        
    return ret, selec_rows

def delete_outlier_rows(dataset):
    
    ret = dataset.copy()
    
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    
    ind_to_keep = []
    
    for i in np.arange(len(dataset[:,0])):
        if(np.count_nonzero(dataset[i, 2:] - mean > 3*std) == 0):
            ind_to_keep.append(i)
    ind_to_keep = np.array(ind_to_keep)
    ind_to_keep = ind_to_keep.astype(int)
    
    return ret[ind_to_keep]

def count_outliers_for_feature(dataset):
    
    ret = dataset.copy()
    
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    
    cnt_vec = np.array([np.count_nonzero(ret[:,i]-mean[i-2] > 3*std[i-2]) for i in np.arange(2,31)] )
    
    return cnt_vec

def subsitute_nan_with_val(dataset, val):
    
    ret = dataset.copy()
    ret[ret==-999] = val
        
    return ret

def subsitute_nan_with_mean(dataset):
    
    ret = dataset.copy()

    for j in range(len(dataset[0,:])):
    
        indexes_no999 = (dataset[:,j] != -999)
        indexes_999 = (dataset[:,j] == -999)
        curr_col_wo999 = ret[indexes_no999,j]
        ret[indexes_999,j] = float(np.mean(curr_col_wo999))
        
    return ret

def subsitute_nan_with_median(dataset):
    
    ret = dataset.copy()

    for j in range(len(dataset[0,:])):
    
        indexes_no999 = (dataset[:,j] != -999)
        indexes_999 = (dataset[:,j] == -999)
        curr_col_wo999 = ret[indexes_no999,j]
        ret[indexes_999,j] = float(np.median(curr_col_wo999))
        
    return ret

def generate_prediction(tx, w):
    
    prediction = tx.dot(w)
    
    index_neg = (prediction < 0)
    index_pos = (prediction >= 0)
    
    prediction[index_neg] = -1
    prediction[index_pos] = 1

    return prediction

def generate_csv(prediction, ids, name):
    
    Id = ids.astype(int)
    Prediction = prediction.astype(int)

    data = [Id, Prediction]
    export_data = zip_longest(*data, fillvalue = '')
    
    with open(name, 'w', encoding="ISO-8859-1", newline='') as file:
        write = csv.writer(file)
        write.writerow(("Id", "Prediction"))
        write.writerows(export_data)

def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})

def feature_regression_cols(y, x, degrees, k_fold, lambdas, seed = 1):
    
    yy = y.copy()
    xx = x.copy()
    
    ind_col = np.arange(yy.size)
    
    y_train, ind_train = delete_nan_elem(yy)
    x_train = xx[ind_train]
    
    ind_train = np.array(ind_train)
    ind_train = ind_train.astype(int)
    
    ind_test = []
    for i in ind_col:
        if np.count_nonzero(ind_train == i) == 0:
            ind_test.append(i)
    ind_test = np.array(ind_test)
    ind_test = ind_test.astype(int)

    y_test = yy[ind_test] # -999 values
    x_test = xx[ind_test]
    
    best_degree, best_lambda, best_rmse = best_degree_selection(x_train, y_train, degrees, k_fold, lambdas, seed)
               
    tx_train = build_poly(x_train, best_degree)

    w = ridge_regression(y_train, tx_train, best_lambda)[0]

    tx_test = build_poly(x_test, best_degree)
    y_test = tx_test.dot(w)
        
    yy[ind_train] = y_train
    yy[ind_test] = y_test
    
    return yy

def feature_regression(ttrain, ttest, degrees, k_fold, lambdas, seed = 1):
    
    train = ttrain.copy()
    n_rows_train = train.shape[0]
    test = ttest.copy()
    n_rows_test = test.shape[0]
    dataset = np.vstack((train, test))
    
    cnt = count_nan_for_feature(dataset)
    
    ind = np.arange(cnt.size)
    
    ind_input = []
    ind_output = []
    for i in ind[2:]:
        if(cnt[i]==0):
            ind_input.append(i)
        else:
            ind_output.append(i)
    
    ind_input = np.array(ind_input)
    ind_input = ind_input.astype(int)
    ind_output = np.array(ind_output)
    ind_output = ind_output.astype(int)
    
    n_input = len(ind_input)
            
    for i in range(len(ind_output)):
        
        yy = feature_regression_cols(dataset[:, ind_output[i]], dataset[:, ind_input[i % n_input]], degrees, k_fold, lambdas, seed)
        dataset[:, ind_output[i]] = yy
                                
    train = dataset[:n_rows_train, :]
    test = dataset[n_rows_train:,:]        
    
    return train, test

def poly_expansion_col_lin(x, y, degrees, k_fold, lambdas, seed = 1):
    
    best_deg = best_degree_selection_x_lin(x, y, degrees, k_fold, lambdas, seed)[0]
    tx = build_poly(x, best_deg)
    
    return tx, best_deg

def poly_expansion_lin(dataset, degrees, k_fold, lambdas, seed = 1):
    
    ret = dataset[:,[0,1]].copy()
    offset_col = np.ones((dataset[:,0].size,1))
    ret = np.hstack((ret, offset_col))
    
    ids, y, tx = split_into_ids_y_tx(dataset)
    degree_selected = np.zeros(tx[0,:].size)
    
    for i in np.arange(len(tx[0,:])):
       exp_cols, degree_selected[i] = poly_expansion_col_lin(tx[:,i], y, degrees, k_fold, lambdas, seed)
       exp_cols = exp_cols[:, 1:]
       ret = np.append(ret, exp_cols, axis=1)
        
    return ret, degree_selected

def poly_expansion_col_log(x, y, degrees, k_fold, lambdas, seed = 1):
    
    best_deg = best_degree_selection_x_log(x, y, degrees, k_fold, lambdas, seed)[0]
    tx = build_poly(x, best_deg)
    
    return tx, best_deg

def poly_expansion_log(dataset, degrees, k_fold, lambdas, seed = 1):
    
    ret = dataset[:,[0,1]].copy()
    offset_col = np.ones((dataset[:,0].size,1))
    ret = np.hstack((ret, offset_col))
   
    ids, y, tx = split_into_ids_y_tx(dataset)
    degree_selected = np.zeros(tx[0,:].size)
    
    for i in np.arange(len(tx[0,:])):
       exp_cols, degree_selected[i] = poly_expansion_col_log(tx[:,i], y, degrees, k_fold, lambdas, seed)
       exp_cols = exp_cols[:, 1:]
       ret = np.append(ret, exp_cols, axis=1)
        
    return ret, degree_selected

def NaN_row_index(dataset, tollerance):
    
    check = []
    for i in range(len(dataset[:,0])):
        check.append( np.count_nonzero(dataset[i,:]==-999))

    check  = np.array(check)   
    ind_nan_r =  check > tollerance
    
    return ind_nan_r

def balance(dataset):
    num_ones = np.sum(dataset[:,1]==1)
    num_minus_ones = np.sum(dataset[:,1]== 0)
    Dim = num_ones + num_minus_ones
    ret = dataset.copy()
    
    index_minus_ones = dataset[:,1] == 0 
    index_ones = dataset[:,1] == 1
    tol = 0
    found = False
    
    if (num_minus_ones > num_ones) :
        diff = num_minus_ones - num_ones
        index = index_minus_ones
    else:
        diff = num_ones - num_minus_ones
        index = index_ones
        
    while not found:
        ind_nan_rows = NaN_row_index(ret, tol)
        to_delete = ind_nan_rows * index
        n = np.count_nonzero(to_delete==True)
        
        if n < diff:
            found = True
        else:
            tol = tol + 1
         
    to_keep = np.array([not to_delete[i] for i in range(len(to_delete))])
    ret = ret[to_keep,:]
        
    index = index[to_keep]
    tol = tol  - 1
    ind_nan = NaN_row_index(ret, tol)
        
    to_del = ind_nan * index
        
    cumsum = np.cumsum(to_del)
    ind = np.min(np.where(cumsum==diff-n))
        
    to_del[ind:] = False
    to_k = np.array([not to_del[i] for i in range(len(to_del))])
    ret = ret[to_k,:]

    return ret

def split_jet(dataset, col24):
    
    #given a dataset, returns a list with the dataset splitted by jet number 
    col_indexes = np.arange(dataset[0,:].size)
    
    row_indexes = np.arange(dataset[:,0].size)
    ind0 = row_indexes[col24==0]
    ind1 = row_indexes[col24==1]
    ind2 = row_indexes[col24==2]
    ind23 = row_indexes[(col24==3) | (col24==2)]
    
    jet_0 = dataset[ind0,:]
    jet_1 = dataset[ind1,:]
    jet_23 = dataset[ind23,:]
    
    
    mylist = []
    mydata = [jet_0, jet_1, jet_23]
    myind = [ind0, ind1, ind23]
    
    for i in range(len(mydata)):
        
        col_index_to_keep = col_indexes[np.std(mydata[i],axis=0) != 0]
        mydata[i] = mydata[i][:,col_index_to_keep]
        internal_list = [mydata[i], myind[i], col_index_to_keep]
     
        mylist.append(internal_list)

    return mylist





def poly_expansion_blind(dataset, deg):
    
    ret = dataset[:,[0,1]].copy()
    offset_col = np.ones((dataset[:,0].size,1))
    ret = np.hstack((ret, offset_col))
        
    for j in range(2, dataset[0,:].size):
        
        curr_expansion = build_poly(dataset[:,j], deg)
        # remove the offset column
        curr_expansion = curr_expansion[:,1:]
        
        ret = np.hstack((ret, curr_expansion))
    
    return ret





##############################################################################
def nan_regression (train, test, degrees, k_fold, lambdas, seed = 1):
    
    n_rows_train = train.shape[0]
    n_rows_test = test.shape[0]
    
    dataset = np.vstack((train, test))
    dataset = np.delete(dataset, 1, axis=0)
    
    cnt = count_nan_for_feature(dataset)
    
    ind = np.arange(cnt.size).astype(int)
    ind_output = 2
        
    y = dataset[:, ind_output].copy()
    tx = dataset[:, ind_output:].copy()
    # ds_expand = poly_expansion_blind(ds, 3)
        
        
    inds = np.arange(y.size)
    y_train, ind_train = delete_nan_elem(y)
    tx_train = tx[ind_train]
        
    ind_train = np.array(ind_train)
    ind_train = ind_train.astype(int)
        
    ind_test = []
    for i in inds:
        if np.count_nonzero(ind_train == i) == 0:
           ind_test.append(i)
    ind_test = np.array(ind_test)
    ind_test = ind_test.astype(int)

    y_test = y[ind_test] # -999 values
    tx_test = tx[ind_test]
        
    best_lambda = cross_validation_demo_tx_lin(y_train, tx_train, k_fold, lambdas)[0]
    w = ridge_regression(y_train, tx_train, best_lambda)[0]

    tx_test = tx[ind_test]
    y_test = tx_test.dot(w)
            
    y[ind_train] = y_train
    y[ind_test] = y_test
        
    dataset[:, ind_output] = y
        
    ttrain = dataset[:n_rows_train, :]
    ttest = dataset[n_rows_train:,:]        
    
    return ttrain, ttest