import numpy as np
from itertools import zip_longest
import csv
from utilities_linear_regression import *

################################ LOADING DATA #################################

def load_train_dataset():
    """ Load training dataset and convert 'b' into 0 and 's' into 1
        Remove the 24th categorical feature
        Return the dataset and the removed feature """
    
    path_dataset = "train.csv"
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)
    first_col = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[0])
    second_col = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1],
                               converters={1: lambda x: 0 if b"b" in x else 1})
    data[:, 0] = first_col
    data[:, 1] = second_col
    col24 = data[:, 24]
    data = np.delete(data, 24, 1)
    
    return data, col24
    
def load_test_dataset():
    """ Load testing dataset and filling the prediction column with 0
        Remove the 24th categorical feature
        Return the dataset and the removed feature"""
        
    path_dataset = "test.csv"
    data = np.genfromtxt(path_dataset, delimiter=",", skip_header=1)
    first_col = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[0])
    second_col = np.genfromtxt(path_dataset, delimiter=",", skip_header=1, usecols=[1],
                               converters={1: lambda x: 0 if b"?" in x else 1})
    data[:, 0] = first_col
    data[:, 1] = second_col
    col24 = data[:, 24]
    data = np.delete(data, 24, 1)
    
    return data, col24

def split_into_ids_y_tx(dataset):
    """Split the dataset into ids, predictions y and model matrix tx
        Args: 
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ids: shape=(N, )
            y: shape=(N, )
            tx: shape=(N, D) """
            
    ids = dataset[:, 0].copy()
    y = dataset[:, 1].copy()
    tx = dataset[:, 2:].copy()
    
    return ids, y, tx

def split_jet(dataset, col24):
    """ Split the dataset according the categorical values of col24 'PRI_jet_num'
        Args:
            dataset: shape=(N, 32) (N number of events)
            col24: shape=(N, ) (24th column of dataset, 22nd feature)
        Return: 
            mylist: list[list_a, list_b, list_c]
                    every internal list refers to a categorical value and contains:
                        - the sub-dataset without all the zero-variance features
                        - the indexes of the events in the sub-dataset
                        - the indexes of the columns in the sub-dataset """
                        
    col_indexes = np.arange(dataset[0,:].size)
    
    row_indexes = np.arange(dataset[:,0].size)
    ind0 = row_indexes[col24==0]
    ind1 = row_indexes[col24==1]
    ind23 = row_indexes[(col24==2) | (col24==3)]
    
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

def generate_csv(prediction, ids, name):
    """ Creates an output file in .csv format for submission to Kaggle or AIcrowd
        Args: 
            predictions (predicted class labels)
            ids (event ids associated with each prediction)
            name (string name of .csv output file to be created)"""
    
    Id = ids.astype(int)
    Prediction = prediction.astype(int)

    data = [Id, Prediction]
    export_data = zip_longest(*data, fillvalue = '')
    
    with open(name, 'w', encoding="ISO-8859-1", newline='') as file:
        write = csv.writer(file)
        write.writerow(("Id", "Prediction"))
        write.writerows(export_data)
        
def optimal_threshold(train, vec_t, lambda_):
    """ Cross validation over a threshold vector vec_t to find the optimal threshold
        (maximization of the accuracy on the training set)
        The best-threshold selection is performed by means of ridge regression 
        Args:
            train: shape=(N, D+2) (N number of events, Dnumber of features)
            vec_t: shape=(101, )
            lambda_: scalar(float) 
        Returns:
            optimal_t: scalar(float) """
            
    N = len(vec_t)
    accuracy = np.zeros(N)
    pred = []
    true = []
    l = len(train)
    
    # training
    for i in range(5):
        perm = np.random.permutation(l)
        train_index = perm[:int(np.floor(l*0.8))]
        test_index = perm[int(np.floor(l*0.8)):]
        train_i = train[train_index,:]
        test_i = train[test_index,:]
        ws = ridge_regression(train_i[:,1], train_i[:,2:], lambda_)[0]
        ys = test_i[:,2:].dot(ws)
        pred.append(ys)
        true.append(test_i[:,1])
        
        
    for i in range(N):
        acc = np.zeros(5)
        for j in range(5):
            vec = np.zeros(len(pred[j]))
            index_neg = (pred[j] < vec_t[i])
            index_pos = (pred[j] >= vec_t[i])
            
            vec[index_neg] = 0
            vec[index_pos] = 1
            
            correct = 0
            for k in range(len(vec)):
                if vec[k] == true[j][k]:
                    correct = correct+1
            accu = correct/len(vec)
            acc[j] = accu
            
        avg = np.mean(acc)
        accuracy[i] = avg
        
    arg = np.where(accuracy == max(accuracy))[0][0]
    optimal_t = vec_t[arg]
    
    return optimal_t

def collect(y_list, index_list, threshold):
    """ Collect and re-order the values of the predictions in y_list according
        to the the indexes in index_list 
        Args:
            y_list: list containing three np.array of non-binary predictions
            index_list: list containing three np.array of row indexes
            threshold: list of three optimal thresholds
        Returns:
            y: shape=(568238, ) array of ordered binary predictions """
            
    N = 568238
    y = np.zeros(N)
    ys = []
    
    for i in range(len(threshold)):
        index_neg = (y_list[i] < threshold[i])
        index_pos = (y_list[i] >= threshold[i])
        yy = np.zeros(len(y_list[i]))
        yy[index_neg] = -1
        yy[index_pos] = 1
        ys.append(yy)
        
    y[index_list[0]] = ys[0]
    y[index_list[1]] = ys[1]
    y[index_list[2]] = ys[2]
        
    return y

def generate_linear_prediction(test, ws):
    """ Given the test set and the weights list, performs the predictions as
        follows : y = tx.dot(w) 
        Args:
            test: shape=(568238, D+2)
            ws: list of three np.array of weights
        Returns:
            ys: list of three np.array non-binary predictions """
    ys = []
    
    for i in range(len(ws)):
        tx = test[i][:, 2:]
        y = tx.dot(ws[i])
        ys.append(y)
        
    return ys

