import numpy as np

from feature_engineering import *
from utilities_linear_regression import *
from utilities_logistic_regression import *
from implementations import *

############################## POST - PROCESSING #############################

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





