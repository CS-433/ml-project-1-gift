import numpy as np
from utilities_linear_regression import *
 

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
        
    return w,loss

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
    
    loss = 0

    for n_iter in range(max_iters):

        for y_batch, tx_batch in batch_iter(y, tx, batch_size=512, num_batches=1):
            # compute a stochastic gradient and loss
            grad, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma * grad
            # calculate loss
            loss = compute_loss(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
    
    return w,loss

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
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    
    return w,loss
   
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
    
    return w,loss

##################################################################################
def log_reg(y, tx, w):
    """return the loss, gradient of the loss, and hessian of the loss.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        loss: scalar number
        gradient: shape=(D, 1) 
        hessian: shape=(D, D)

   
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    
    return loss, grad, hess


def reg_log_reg(y, tx, w, lambda_):
    """return the loss and gradient.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        lambda_: scalar

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)

    
    """
    pen_log_loss = calculate_loss(y,tx,w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_gradient(y,tx,w)
    pen = (2*lambda_*w)
    pen_grad = grad + pen
    
    return pen_log_loss, pen_grad

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss

    
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    txw = tx.dot(w)
    log_loss = np.mean(np.log(1 + np.exp(txw)) - y*txw)
    
    return log_loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss.
    
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a vector of shape (D, 1)

    
    """
    # ***************************************************
    N = y.size
    grad = (1/N) * np.dot(tx.T, sigmoid(tx.dot(w))-y)

    return grad

def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 
        gamma: float

    Returns:
        loss: scalar number
        w: shape=(D, 1) 

   """
    # ***************************************************
    loss = calculate_loss(y, tx, w)
    w = w - gamma*calculate_gradient(y, tx, w)
    
    return loss, w


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: scalar
        lambda_: scalar

    Returns:
        loss: scalar number
        w: shape=(D, 1)

   
    """
    # ***************************************************
    loss, grad = reg_log_reg(y, tx, w, lambda_)
    w = w - gamma*grad
    
    return loss, w
   
   
   
   
   
