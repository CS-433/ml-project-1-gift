import numpy as np
import matplotlib.pyplot as plt
from implementations import * 

######################### UTILITIES LINEAR REGRESSION ########################
def ridge_regression(y, tx, lambda_):
    """ The Ridge Regression algorithm (RR)
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            lambda_: scalar(float) penalization parameter
        Returns:
            w: shape=(D, ) optimal weights
            loss: scalar(float) """
            
    N = y.size
    D = tx[0,:].size
    I = np.identity(D)
    A = np.dot(tx.T,tx) + (lambda_*2*N)*I
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A, b)
    
    loss = compute_loss(y, tx, w)
    
    return w, loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and
        the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching 
        elements from `y` and `tx`. Data can be randomly shuffled to avoid 
        ordering in the original data messing with the randomness of the minibatches.
        Example of use :
            for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
                <DO-SOMETHING> """
    
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

    return

def compute_mse(y, tx, w):
    """ Calculate the mse for the vector e = y - tx.dot(w) """
    
    return 1 / 2 * np.mean((y-tx.dot(w)) ** 2)

def compute_mae(y, tx, w):
    """ Calculate the mae for vector e = y - tx.dot(w) """
    
    return np.mean(np.abs(y-tx.dot(w)))

def compute_loss(y, tx, w):
    """ Calculate the loss using either MSE or MAE 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2,). The vector of model parameters.
        Returns:
            loss: scalar(float), corresponding to the input parameters w """
    
    return compute_mse(y, tx, w)

def compute_gradient(y, tx, w):
    """ Compute the gradient at w 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2, ). The vector of model parameters.
        Returns:
            grad: shape=(D, ) (gradient of the loss at w) """
            
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    
    return grad

def compute_stoch_gradient(y, tx, w):
    """ Compute a stochastic gradient 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2, ). The vector of model parameters.
        Returns:
            grad: shape=(D, ) (stochastic gradient of the loss at w)"""
    
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    
    return grad

def build_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree """
    phi_x = np.zeros((x.size, degree+1))
    for i in range(len(x)):
        curr_row = np.array([x[i]**deg for deg in range(degree+1)])
        phi_x[i] = curr_row
        
    return phi_x

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold 
        Args:
            y: shape=(N,)
            k_fold: K in K-fold, i.e. the fold num
            seed: the random seed
        Returns:
            ret: shape=(k_fold, N/k_fold) with the data indices for each fold """
    num_row = y.shape[0]
    interval = int(num_row / k_fold) # Here it computes the number of intervals
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation_x_lin(y, x, k_indices, k, lambda_, degree):
    """ Return the loss of ridge regression for a fold corresponding to k_indices 
        The feature x gets expanded by a degree to build the regression matrix tx 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            k_indices: 2D array returned by build_k_indices()
            k: scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
            lambda_: scalar, cf. ridge_regression()
            degree: scalar, cf. build_poly()
        Returns:
            loss_tr: scalar(float), rmse = sqrt(2 mse) of the training set
            loss_te: scalar(float), rmse = sqrt(2 mse) of the testing set """
            
    # get k'th subgroup in test, others in trai
    k_fold = len(k_indices)
    
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    tx_test = build_poly(x_test, degree)
    
    ind_train = []
    ind_train = np.append(ind_train, k_indices[np.arange(k_fold)!=k])
    ind_train = np.array(ind_train)
    ind_train = ind_train.astype(int)
                 
    y_train = y[ind_train]
    x_train = x[ind_train]
    tx_train = build_poly(x_train, degree)
    
    # ridge regression
    w_k = ridge_regression(y_train, tx_train, lambda_)[0]
    
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*compute_mse(y_train, tx_train, w_k))
    loss_te = np.sqrt(2*compute_mse(y_test, tx_test, w_k))
    
    return loss_tr, loss_te

def cross_validation_tx_lin(y, tx, k_indices, k, lambda_):
    """ Return the loss of ridge regression for a fold corresponding to k_indices
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            k_indices: 2D array returned by build_k_indices()
            k: scalar, the k-th fold
            lambda_: scalar, cf. ridge_regression()
        Returns:
            loss_tr: scalar(float), rmse = sqrt(2 mse) of the training set
            loss_te: scalar(float), rmse = sqrt(2 mse) of the testing set """

    # get k'th subgroup in test, others in train
    k_fold = len(k_indices)
    
    y_test = y[k_indices[k]]
    tx_test = tx[k_indices[k],:]
    
    ind_train = []
    ind_train = np.append(ind_train, k_indices[np.arange(k_fold)!=k])
    ind_train = [int(ind_train[i]) for i in range(len(ind_train))]
                 
    y_train = y[ind_train]
    tx_train = tx[ind_train,:]
    
    # ridge regression
    w_k = ridge_regression(y_train, tx_train, lambda_)[0]
    
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*compute_mse(y_train, tx_train, w_k))
    loss_te = np.sqrt(2*compute_mse(y_test, tx_test, w_k))
    
    return loss_tr, loss_te

def cross_validation_demo_x_lin(x, y, degree, k_fold, lambdas):
    """ Cross validation over regularisation parameter lambda
        The feature x gets expanded by a degree to build the regression matrix tx 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degree: integer, degree of the polynomial expansion
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda """
    
    seed = 12
    
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    
    for l in lambdas:
        loss_tr_sum = 0
        loss_te_sum = 0
        for k in range(k_fold):
            loss_tr, loss_te = cross_validation_x_lin(y, x, k_indices, k, l, degree)
            loss_tr_sum += loss_tr
            loss_te_sum += loss_te
        rmse_tr = np.append(rmse_tr, loss_tr_sum/k_fold)
        rmse_te = np.append(rmse_te, loss_te_sum/k_fold)
    
    best_ind = np.argmin(rmse_te)
    best_lambda = lambdas[best_ind]
    best_rmse = rmse_te[best_ind]
    
    return best_lambda, best_rmse

def cross_validation_demo_tx_lin(y, tx, k_fold, lambdas):
    """ Cross validation over regularisation parameter lambda 
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            degree: integer, degree of the polynomial expansion
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda """
    
    seed = 1
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for l in lambdas:
        r_tr = []
        r_te = []
    
        for k in range(k_fold):  # we do this to perform the training using all the data
            r_tr.append(cross_validation_tx_lin(y, tx, k_indices, k, l)[0])  
            r_te.append(cross_validation_tx_lin(y, tx, k_indices, k, l)[1]) 
        rmse_tr.append(np.mean(r_tr)) # mean of the rmse test for each of the considered lambda
        rmse_te.append(np.mean(r_te))
    
    best_rmse = min(rmse_te)
    minpos = rmse_te.index(best_rmse)  # Get the position of the minimum value of average loss 
    best_lambda = lambdas[minpos]
    
    return best_lambda, best_rmse, rmse_tr, rmse_te

def best_degree_selection_x_lin(x, y, degrees, k_fold, lambdas, seed = 1):
    """ Cross validation over regularisation parameter lambda and degree 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degrees: shape = (d,), where d is the number of degrees to test 
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_degree: scalar, value of the best degree
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda """
            
    # define lists to store the loss of training data and test data
    best_lambdas = []
    best_rmses = []
    
    for deg in degrees:
        
        best_lambdas.append(cross_validation_demo_x_lin(x, y, deg, k_fold, lambdas)[0])
        best_rmses.append(cross_validation_demo_x_lin(x, y, deg, k_fold, lambdas)[1])
    
    best_ind = np.argmin(best_rmses)
    best_degree = degrees[best_ind]
    best_lambda = best_lambdas[best_ind]
    best_rmse = best_rmses[best_ind]
    
    return best_degree, best_lambda, best_rmse

def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")