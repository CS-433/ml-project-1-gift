import numpy as np

from loading_data import *
from preprocessing import *
from cleaning_dataset import *
from utilities_linear_regression import *
from utilities_logistic_regression import *
from implementations import *
from postprocessing import *

############################# FEATURE ENGINEERING ############################

def standardize_train(train):
    """Standardize the training dataset
       Args:
           train: shape=(N, D+2) (N number of events, D number of features)
       Returns:
           ret: shape=(N, D+2) """
               
    ret = train.copy()
    mean = np.mean(ret[:,2:], axis=0)
    std = np.std(ret[:,2:], axis=0)
    ret[:,2:] = (ret[:,2:] - mean)/std
    
    return ret, mean, std

def standardize_test(test, mean_train, std_train):
    """ Standardize the testing dataset, upon the mean and the standard deviation 
        of training dataset 
        Args:
            test: shape=(N, D+2) (N number of events, D number of features)
            mean_train: shape=(D+2, )
            std_train: shape=(D+2, )
        Returns:
            ret: shape=(N, D+2) """
                
    ret = test.copy()
    ret[:,2:] = (ret[:,2:] - mean_train)/std_train
    
    return ret

def impute_nan_with_median(dataset):
    """ Impute the -999 values with the median of the corresponding feature
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
        Returns:
            ret: shape=(N, D+2) """
            
    ret = dataset.copy()

    for j in range(len(dataset[0,:])):
    
        indexes_no999 = (dataset[:,j] != -999)
        indexes_999 = (dataset[:,j] == -999)
        curr_col_wo999 = ret[indexes_no999,j]
        ret[indexes_999,j] = float(np.median(curr_col_wo999))
        
    return ret

def feature_regression_cols(y, x, degrees, k_fold, lambdas, seed = 1):
    """ Polynomial regression of y starding from x
        The best-degree polynomial expansion of x is performed by means of ridge
        regression cross-validation over a vector of degrees and lambdas:
        The return value yy is the full y, where -999 values are filled with the
        corresponding prediction
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            yy: shape=(N, ) """
            
    yy = y.copy()
    xx = x.copy()
    
    ind_col = np.arange(yy.size)
    
    y_train, ind_train = remove_nan_elem(yy)
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
    """ Polynomial regression of the -999 values of training and testing dataset
        The i-th feature to fill is predicted by the i-th full feature
        The internal regression in performed with the function:
            feature_regression_cols(y ,x, ...) 
        The return valuest train and test are the 'full' training and testing
        sets, where -999 values are filled with the corresponding prediction
        Args:
            ttrain: shape=(N, D+2) (N number of events in the training set)
            ttest: shape=(M, D+2) (M number of events in the testing set)
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            train: shape=(N, D+2)
            test: shape=(N, D+2) """
    
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

def feature_regression_col3(ttrain, ttest, k_fold, lambdas, seed = 1):
    """ Linear regression of the -999 values of the 1st feature 
        (3rd column) of the training (ttrain) and testing (ttest) dataset
        The regression is performed by ridge regression, where the penalizing
        parameter is chosen with cross validation 
        Args:
            ttrain: shape=(N, D+2) (N number of events in the training set)
            ttest: shape=(M, D+2) (M number of events in the testing set)
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            train: shape=(N, D+2)
            test: shape=(N, D+2) """
            
    col_train = ttrain[:,:2].copy()
    train = ttrain[:,2:].copy()
    n_rows_train = train.shape[0]
    
    col_test = ttest[:,:2].copy()
    test = ttest[:,2:].copy()
    n_rows_test = ttest.shape[0]
    
    dataset = np.vstack((train, test))
        
    index_test = dataset[:,0]==-999  # rows with nan
    index_train = dataset[:,0]!=-999  # rows without nan
    train_reg = dataset[index_train,:]
    test_reg = dataset[index_test,:]
    y = train_reg[:,0]
    tx = train_reg[:,1:]
    
    lambda_= cross_validation_demo_tx_lin(y, tx, k_fold, lambdas)[0]
    # print(lambda_)
    w = ridge_regression(y, tx, lambda_)[0]
    yy = test_reg[:,1:].dot(w)
    
    dataset[index_test, 0] = yy
    
    train = np.c_[col_train, dataset[:n_rows_train,:]]
    test = np.c_[col_test, dataset[n_rows_train:,:]]

    return train, test

def cosine(dataset, angle_cols):
    """ Correct the angle_cols features of the dataset by performing
        a cosine tranformation f(x) = cos(x)
        (angle_cols is an arry containing the indexes of the columns to be tranformed)
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
            angle_cols: shape=(n, ) (n number of features representing angles)
        Returns:
            ret: shape=(N, D+2) """
            
    ret = dataset.copy()
    ret[:, angle_cols] = np.cos(ret[:, angle_cols])
    
    return ret   

def correct_skewness(dataset, skew_cols):
    """ Correct the skew_cols features of the dataset by performing
        a logaritmic tranformation f(x) = log(1+x)
        The features to correct are skewned and non-negative
        (skew_cols is an arry of bools whose True values are the columns to be tranformed)
        Args:
            dataset: shape=(N, D+2) (N number of events, D number of features)
            skew_cols: shape=(N, )
        Returns:
            ret: shape=(N, D+2) """
            
    ret = dataset.copy()
    ret[:, skew_cols] = np.log(1 + ret[:,skew_cols])
        
    return ret

def poly_expansion_col_lin(x, y, degrees, k_fold, lambdas, seed = 1):
    """ Polynomial expansion of the column x to predict y
        The best-degree selection is performed by means of ridge regression
        cross-validation over a vector of degrees and lambdas
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_deg: scalar(int)
            tx: shape=(N, best_deg+1) """
            
    best_deg = best_degree_selection_x_lin(x, y, degrees, k_fold, lambdas, seed)[0]
    tx = build_poly(x, best_deg)
    tx = tx[:,1:]
    
    return tx, best_deg

def poly_expansion_lin(dataset, degrees, k_fold, lambdas, seed = 1):
    """ Polynomial expansion of design matrix tx of the dataset
        The best-degree selection for each feature is performed by the function:
            poly_expansion_col_lin(x, y, ...) 
        Args:
            dataset: shape=(N, D+2) (N number of events)
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            degree_selected: shape=(D, )
            ret: shape=(N, D*sum(degree_selected)+1) """
            
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
    """ Polynomial expansion of the column x to predict y
        The best-degree selection is performed by means of a penalized logistic 
        regression with gradient descent cross-validation over a vector of 
        degrees and lambdas 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_deg: scalar(int)
            tx: shape=(N, best_deg+1)"""
    best_deg = best_degree_selection_x_log(x, y, degrees, k_fold, lambdas, seed)[0]
    tx = build_poly(x, best_deg)
    tx = tx[:,1:]
    
    return tx, best_deg

def poly_expansion_log(dataset, degrees, k_fold, lambdas, seed = 1):
    """ Polynomial expansion of design matrix tx of the dataset
        The best-degree selection for each feature is performed by the function:
            poly_expansion_col_log(x, y, ...)
        Args:
            dataset: shape=(N, D+2) (N number of events)
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            degree_selected: shape=(D, )
            ret: shape=(N, D*sum(degree_selected)+1) """
    ret = []
    ret = np.array(ret)
    ids, y, tx = split_into_ids_y_tx(dataset)
    degree_selected = np.zeros(tx[0,:].size)
    
    for i in np.arange(len(tx[0,:])):
        if i == 0:
            ret, degree_selected[i] = poly_expansion_col_log(tx[:,i], y, degrees, k_fold, lambdas, seed)
        else:
            exp_cols, degree_selected[i] = poly_expansion_col_log(tx[:,i], y, degrees, k_fold, lambdas, seed)
            ret = np.append(ret, exp_cols, axis=1)
        
    return ret, degree_selected

def best_degree_dataset(dataset, degrees, k_fold, lambdas, seed = 1):
    """ Cross validation over degree to find the optimal polynomial expansion
        degree of the model matrix tx.
        The best-degree selection is performed by means of ridge regression
        cross-validation over a vector of degrees and lambdas 
        Args:
            dataset: shape=(N, D+2) (N number of events)
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_degree: scalar(int)
            best_lambda: scalar(float)
            best_rmse: scalar(float) """
    
    ids, y, tx = split_into_ids_y_tx(dataset)
    # define lists to store the loss of training data and test data
    best_lambdas = []
    best_rmses = []
    
    for deg in degrees:
        
        best_lambdas.append(cross_validation_demo_deg_tx_lin(y,tx, deg, k_fold, lambdas)[0])
        best_rmses.append(cross_validation_demo_deg_tx_lin(y,tx, deg, k_fold, lambdas)[1])
    
    best_ind = np.argmin(best_rmses)
    best_degree = degrees[best_ind]
    best_lambda = best_lambdas[best_ind]
    best_rmse = best_rmses[best_ind]
    
    return best_degree, best_lambda, best_rmse

def poly_expansion_blind(dataset, deg):
    """ Polynomial expansion of design matrix tx of the dataset
        Each feature gets expanded by a degree deg
        Args:
            dataset: shape=(N, D+2) (N number of events)
            deg: scalar(int)
        Returns:
            ret: shape=(N, D*deg+1) """
            
    ret = dataset[:,[0,1]].copy()
    offset_col = np.ones((dataset[:,0].size,1))
    ret = np.hstack((ret, offset_col))
        
    for j in range(2, dataset[0,:].size):
        
        curr_expansion = build_poly(dataset[:,j], deg)
        # remove the offset column
        curr_expansion = curr_expansion[:,1:]
        
        ret = np.hstack((ret, curr_expansion))
    
    return ret

def poly_expansion_blind_degrees(dataset, degrees):
    """ Polynomial expansion of design matrix tx of the dataset
        Each feature gets expanded by the corresponding degree in degrees
        Args:
            dataset: shape=(N, D+2) (N number of events)
            degrees: shape=(D, )
        Returns:
            ret: shape=(N, D*sum(degrees)+1) """
            
    ret = dataset[:,[0,1]].copy()
    offset_col = np.ones((dataset[:,0].size,1))
    ret = np.hstack((ret, offset_col))
    degrees = degrees.astype(int)
        
    for j in range(2, dataset[0,:].size):
        
        curr_expansion = build_poly(dataset[:,j], degrees[j-2])
        # remove the offset column
        curr_expansion = curr_expansion[:,1:]
        
        ret = np.hstack((ret, curr_expansion))
    
    return ret

def feature_cross_products(dataset):
    """ Given a dataset with D features (with the offset as first column) 
        Returns a dataset obtained by the cross-product of D-1 coupled features
        (every feature, but the offset, gets multiplied by all the others 
         Args:
             dataset: shape=(N, D+2) (N number of events)
         Returns:
             prod_cols: shape=(N, p) (p number of cross products) """
             
    sel_ds =  dataset[:,3:]
    length_col = sel_ds.shape[1]
    lenght_row = sel_ds.shape[0]
    prod_cols = np.zeros((lenght_row, 1))
    
    for i in np.arange(length_col):
        p_col = sel_ds[:,(i+1):]
        p_i =  sel_ds[:,i].reshape(lenght_row ,1)
        prod_cols = np.c_[prod_cols , p_i*p_col ]
    
    return prod_cols[:, 1:]

def squareroot(dataset):
    """ Given a dataset with D features (with the offset as first column)
        Returns a dataset obtained by performing the square root of the D-1
        rescaled features (not the offset) 
        Args:
            dataset: shape=(N, D+2) (N number of events)
        Returns:
            prod_cols: shape=(N, D-1) """
            
    maximum = np.max(dataset,axis=0)
    minimum = np.min(dataset,axis=0)
    ret = (dataset[:,3:] - minimum[3:])/(maximum[3:] - minimum[3:])
    
    ret = np.sqrt(ret)
    
    return ret
