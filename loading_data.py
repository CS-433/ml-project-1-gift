import numpy as np
from itertools import zip_longest
import csv

############################# LOADING DATA ###################################

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
