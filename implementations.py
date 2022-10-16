import numpy as np
import matplotlib.pyplot as plt
import doctest
import io
import sys
from utilities import *


#%%
def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        loss: loss value (scalar) for the last iteration of GD
        w: model parameters as numpy arrays of shape (D, ), for the last iteration of GD
    """
    
    w = initial_w
    for n_iter in range(max_iters):
        
        # compute gradient
        grad, err = compute_gradient(y, tx, w)
        
        # update w by gradient descent
        w = w - gamma * grad
        
        # compute loss
        err = y - tx.dot(w)
        loss = calculate_mse(err)
        
        print("GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss


#%%
def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """

    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            
            # update w through the stochastic gradient update
            w = w - gamma * grad
            
            # calculate loss
            loss = compute_loss(y, tx, w)
            
        print("SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
            bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss


#%%
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns loss, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.

    """
    
    # solve the normal equations
    A = np.dot(tx.T,tx)
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A, b)
    
    # compute loss
    loss = compute_loss(y, tx, w)
    
    return w, loss
    
#%%
def ridge_regression(y, tx, lambda_):
    """implement ridge regression.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.

    """
    
    N = y.size
    D = tx[0,:].size
    I = np.identity(D)
    A = np.dot(tx.T,tx)+(lambda_*2*N)*I
    b = np.dot(tx.T,y)
    
    # solve the ridge linear system
    w = np.linalg.solve(A, b)
    
    # compute loss
    loss = compute_loss(y, tx, w)
    
    return w, loss

# Cross-Validation functions

# Build the indixes used to divide the sample into k subsamples

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.
    
    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold

    >>> build_k_indices(np.array([1., 2., 3., 4.]), 2, 1)
    array([[3, 2],
           [0, 1]])
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold) # Here it computes the number of intervals
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

# Execute the learning process on the (k-1) blocks, keeping one of the k blocks as test set

def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test root mean square errors rmse = sqrt(2 mse)

    >>> cross_validation(np.array([1.,2.,3.,4.]), np.array([6.,7.,8.,9.]), np.array([[3,2], [0,1]]), 1, 2, 3)
    (0.019866645527597114, 0.33555914361295175)
    """
    
    y_test = y[k_indices[k]]
    x_test = x[k_indices[k]]
    y_train = y[k_indices[np.arange(len(k_indices))!=k]] 
    y_train = y_train.reshape(np.prod(y_train.shape)) # we need to reshape it to make it again a vector
    x_train = x[k_indices[np.arange(len(k_indices))!=k]]
    x_train = x_train.reshape(np.prod(x_train.shape)) # reshape also here
    # get k'th subgroup in test, others in train: TODO
    # ***************************************************
    
    # ***************************************************
    x_poly_test = build_poly(x_test,degree)
    x_poly_train = build_poly(x_train,degree)
    # form data with polynomial degree: TODO
    # ***************************************************

    # ***************************************************
    w_ridge = ridge_regression(y_train,x_poly_train , lambda_)
    # ridge regression: TODO
    # ***************************************************
    N_train = len(y_train)
    N_test = len(y_test)
    
    # ***************************************************
    loss_tr = (1/(2*N_train))*np.sum((y_train - x_poly_train.dot(w_ridge))**2)
    loss_te = (1/(2*N_test))*np.sum((y_test - x_poly_test.dot(w_ridge))**2)
    # calculate the loss for train and test data: TODO
    # ***************************************************
    
    return np.sqrt(2*loss_tr),np.sqrt(2*loss_te)

# Execute the learning process using all the data and keeping one of the blocks as test set at each iteration


from utilities import cross_validation_visualization

def cross_validation_demo(degree, k_fold, lambdas):
    """cross validation over regularisation parameter lambda.
    
    Args:
        degree: integer, degree of the polynomial expansion
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda
        best_rmse : scalar, the associated root mean squared error for the best lambda
    """
    
    seed = 1 # set the seed like the one of the best_degree_selection function; put 12 for the graph
    degree = degree
    k_fold = k_fold
    lambdas = lambdas
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for l in lambdas:
        r_tr = []
        r_te = []
    
    # ***************************************************
    
        for k in range(k_fold):  # we do this to perform the training using all the data
            r_tr.append(cross_validation(y, x, k_indices, k, l, degree)[0])  
            r_te.append(cross_validation(y, x, k_indices, k, l, degree)[1]) 
        rmse_tr.append(np.mean(r_tr)) # mean of the rmse test for each of the considered lambda
        rmse_te.append(np.mean(r_te))
    # cross validation over lambdas: TODO
    # ***************************************************
    best_rmse = min(rmse_te)
    minpos = rmse_te.index(best_rmse)  # Get the position of the minimum value of average loss 
    best_lambda = lambdas[minpos]
    #cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    #print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    return best_lambda, best_rmse


