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
    """ Load training dataset and convert 'b' in 0 and 's' in 1
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
    """ Split the dataset into ids, y and tx
        ids: (N,)
        y:   (N,)
        tx:  (N,D)"""
    ids = dataset[:, 0].copy()
    y = dataset[:, 1].copy()
    tx = dataset[:, 2:].copy()
    
    return ids, y, tx

def split_jet(dataset, col24):
    """ Split the dataset according the categorical values of col24 'PRI_jet_num'
        Return: a list whose elements are a list containing
                the sub-dataset without all the zero-variance features
                the indexes of the events in the sub-dataset
                the indexes of the columns in the sub-dataset"""
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

def collect(y_list, my_list):
    """ Collect and re-order the values of the predictions in y_list according
        to the the indexes in my_list[i][1] (see split_jet as a reference) """
    N = 568238
    y = np.zeros(N)
    y[my_list[0][1]] = y_list[0]
    y[my_list[1][1]] = y_list[1]
    y[my_list[2][1]] = y_list[2]

    index_neg = (y < 0.5)
    index_pos = (y >= 0.5)

    ys = y
    ys[index_neg] = -1
    ys[index_pos] = 1

    return ys