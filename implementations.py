import numpy as np
from utilities_linear_regression import *
from utilities_logistic_regression import * 

############################# IMPLEMENTATIONS ################################
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """ The Gradient Descent (GD) algorithm 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            initial_w: shape=(D, ). The initial guess for the model parameters
            max_iters: total number of iterations of GD (scalar(int))
            gamma: stepsize (scalar)
        Returns:
            loss: loss value (scalar) of the last iteration of GD
            w: model parameters as numpy arrays of shape (D, ), the last
                iteration of GD """

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y,tx,initial_w)]
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad, err = compute_gradient(y, tx, w)
        loss = 1/2 * np.mean(err**2)
        # update w by gradient descent
        w = w - gamma * grad
        # store w and loss
        ws.append(w)
        losses.append(loss)
        
    return loss, w

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """ The Stochastic Gradient Descent algorithm (SGD) 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            initial_w: shape=(D, ). The initial guess for the model parameters
            batch_size: a scalar denoting the number of data points in a 
                        mini-batch used for computing the stochastic gradient
            max_iters: total number of iterations of GD (scalar(int))
            gamma: stepsize (scalar)
        Returns:
            loss: loss value (scalar) of the last iteration of SGD
            w: model parameters as numpy arrays of shape (D, ), the last
                iteration of SGD"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w)]
    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=8, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    
    return loss, w

def least_squares(y, tx):
    """ The Least Squares algorithm (LS) 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
        Returns:
            w: shape=(D, ) optimal weights
            mse: scalar(float) """
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    mse = compute_mse(y, tx, w)
    
    return w, mse

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


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """ Logistic Regression by Gradient Descent algorithm 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            initial_w: shape=(D, ). The initial guess for the model parameters
            batch_size: a scalar denoting the number of data points in a 
                        mini-batch used for computing the stochastic gradient
            max_iters: total number of iterations of GD (scalar(int))
            gamma: stepsize (scalar)
        Returns:
            loss: loss value (scalar) of the last iteration of GD
            w: model parameters as numpy arrays of shape (D, ), the last
                iteration of GD"""
    
    # init parameters
    threshold = 1e-8
    losses = []

    w = initial_w

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """ Penalized Logistic Regression by Gradient Descent algorithm 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            lambda_: scalar(float) penalization parameter
            initial_w: shape=(D, ). The initial guess for the model parameters
            batch_size: a scalar denoting the number of data points in a 
                        mini-batch used for computing the stochastic gradient
            max_iters: total number of iterations of GD (scalar(int))
            gamma: stepsize (scalar)
        Returns:
            loss: loss value (scalar) of the last iteration of GD
            w: model parameters as numpy arrays of shape (D, ), the last
                iteration of GD """
    
    threshold = 1e-3
    losses = []

    w = initial_w
    loss = 0
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w
