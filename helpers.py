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

def poly_expansion_lin_train(dataset, degrees, k_fold, lambdas, seed = 1):
    
    ret = []
    ret = np.array(ret)
    ids, y, tx = split_into_ids_y_tx(dataset)
    degree_selected = np.zeros(tx[0,:].size)
    
    for i in np.arange(len(tx[0,:])):
        if i == 0:
            ret, degree_selected[i] = poly_expansion_col_lin(tx[:,i], y, degrees, k_fold, lambdas, seed)
        else:
            exp_cols, degree_selected[i] = poly_expansion_col_lin(tx[:,i], y, degrees, k_fold, lambdas, seed)
            ret = np.append(ret, exp_cols, axis=1)
        
    return ret, degree_selected

def poly_expansion_col_log(x, y, degrees, k_fold, lambdas, seed = 1):
    
    best_deg = best_degree_selection_x_log(x, y, degrees, k_fold, lambdas, seed)[0]
    tx = build_poly(x, best_deg)
    
    return tx, best_deg

def poly_expansion_log(dataset, degrees, k_fold, lambdas, seed = 1):
    
    ret = []
    ret = np.array(ret)
    ids, y, tx = split_into_ids_y_tx(dataset)
    
    for i in np.arange(len(tx[0,:])):
        
        tx = poly_expansion_col_log(tx[:,i], y, degrees, k_fold, lambdas, seed)
        ret = np.append(ret, tx)
        
    return ret

