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

