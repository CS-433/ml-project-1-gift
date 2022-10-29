import numpy as np
from utilities_linear_regression import *
from utilities_logistic_regression import * 

############################# IMPLEMENTATIONS ################################

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

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """ The Gradient Descent (GD) algorithm 
        Args:
            y: shape=(N, )
            tx: shape=(N, 2)
            initial_w: shape=(2, ). The initial guess for the model parameters
            max_iters: total number of iterations of GD (scalar(int))
            gamma: stepsize (scalar)
        Returns:
            losses: a list of length max_iters containing the loss value 
                    (scalar) for each iteration of GD
            ws: a list of length max_iters containing the model parameters 
                as numpy arrays of shape (2, ), for each iteration of GD """

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

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
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
            losses: a list of length max_iters containing the loss value 
                    (scalar) for each iteration of SGD
            ws: a list of length max_iters containing the model parameters 
                as numpy arrays of shape (2, ), for each iteration of SGD"""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = [compute_loss(y, tx, initial_w)]
    w = initial_w

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
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

def learning_by_gradient_descent(y, tx, w, gamma):
    """ Do one step of gradient descent using logistic regression. 
        Return the loss and the updated w 
        Args:
            y:  shape=(N, 1) (N number of events)
            tx: shape=(N, D) (D number of features)
            w:  shape=(D, 1) 
            gamma: float
        Returns:
            loss: scalar number
            w: shape=(D, ) """
    
    loss = calculate_loss(y, tx, w)
    w = w.reshape(-1)
    w = w - gamma*calculate_gradient(y, tx, w)
    
    return loss, w

def logistic_regression_gradient_descent_demo(y, x):
    """ Logistic Regression by Gradient Descent algorithm """
    
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.5
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w

def learning_by_newton_method(y, tx, w, gamma):
    """ Do one step of Newton's method
        Return the loss and updated w 
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
            gamma: scalar
    Returns:
            loss: scalar number
            w: shape=(D, 1) """
            
    loss, grad, hess = logistic_regression(y, tx, w)
    g = np.linalg.solve(hess,grad)
    w = w - gamma*g
   
    return loss, w

def logistic_regression_newton_method_demo(y, x):
    """ Logistic Regression by Newton Method algorithm """
    # init parameters
    max_iter = 100
    threshold = 1e-8
    lambda_ = 0.1
    gamma = 1.
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """ Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w 
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
            gamma: scalar
            lambda_: scalar
        Returns:
            loss: scalar number
            w: shape=(D, 1) """
            
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w.reshape(-1)
    w = w - gamma*grad
    
    return loss, w

def logistic_regression_penalized_gradient_descent_demo(y, tx, lambda_):
    """ Penalized Logistic Regression by Gradient Descent algorithm """
    
    # init parameters
    max_iter = 10000
    gamma = 0.5
    threshold = 1e-3
    losses = []

    # build tx
    # tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))
    
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        #if iter % 100 == 0:
        #    print("Current iteration={i}, loss={l}".format(i=iter, l=loss))    
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return loss, w