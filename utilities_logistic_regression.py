import numpy as np

######################### UTILITIES LOGISTIC REGRESSION #######################
def learning_by_gradient_descent(y, tx, w, gamma):
    """ Do one step of gradient descent using logistic regression. 
        Return the loss and the updated w.
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1) 
            gamma: float
        Returns:
            loss: scalar number
            w: shape=(D, 1) """
    
    loss = calculate_loss(y, tx, w)
    w = w.reshape(-1)
    w = w - gamma*calculate_gradient(y, tx, w)
    
    return loss, w

def logistic_regression_gradient_descent_demo(y, x):
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
        # log info
        #if iter % 100 == 0:
         #   print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    # visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent", True)
    # print("loss={l}".format(l=calculate_loss(y, tx, w)))
    
    return w

def logistic_regression_stochastic_gradient_descent_demo(y, x, batch_size=1):
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
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            
            # get loss and update w.
            loss, w = learning_by_gradient_descent(y_batch, tx_batch, w, gamma)
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    
    return w

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """ Do one step of gradient descent, using the penalized logistic regression.
        Return the loss and updated w.
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
            gamma: scalar
            lambda_: scalar
        Returns:
            loss: scalar number
            w: shape=(D, 1)"""
   
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w.reshape(-1)
    w = w - gamma*grad
    
    return loss, w

def logistic_regression_penalized_gradient_descent_demo(y, tx, lambda_):
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
    
    return w

def logistic_regression_penalized_stochastic_gradient_descent_demo(y, tx, lambda_, batch_size=1):
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
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # get loss and update w.
            loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
            losses.append(loss)
            if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
                break
    return w


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
        
def sigmoid(t):
    """ Apply sigmoid function on t """
    return (1 + np.exp(-t))**(-1)

def calculate_loss(y, tx, w):
    """ Compute the cost by negative log likelihood
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2,). The vector of model parameters.
        Returns:
            loss: scalar(float), corresponding to the input parameters w """
            
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]

    txw = tx.dot(w)
    ttxw = txw.reshape(-1)
    log_loss = np.mean(np.log(1 + np.exp(ttxw)) - y*ttxw)
    
    return log_loss

def calculate_gradient(y, tx, w):
    """ Compute the gradient of logistic loss 
        Args:
            y: shape=(N, )
            tx: shape=(N, D)
            w: shape=(2, ). The vector of model parameters.
        Returns:
            grad: shape=(D, ) (gradient of the loss at w) """
            
    N = y.size
    w = w.reshape(-1)
    grad = (1/N) * np.dot(tx.T, sigmoid(tx.dot(w))-y)

    return grad

def calculate_hessian(y, tx, w):
    """ Return the Hessian of the logistic loss function 
        Args:
            y:  shape=(N, 1) ( N number of events)
            tx: shape=(N, D) (D number of features)
            w:  shape=(D, 1) 
        Returns:
            hess: shape=(D, D), hessian matrix """
    
    N = y.size
    diagonal = np.ndarray.flatten(sigmoid(tx.dot(w))*(1-sigmoid(tx.dot(w))))
    S = np.diag(diagonal)
    hess = (1/N) * np.dot(np.dot(tx.T,S),tx)
    
    return hess

def logistic_regression(y, tx, w):
    """ Return the logistic loss, gradient of the logistic loss, and hessian 
        of the logistic loss 
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1) 
        Returns:
            loss: scalar number
            gradient: shape=(D, 1) 
            hessian: shape=(D, D) """
            
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    hess = calculate_hessian(y, tx, w)
    
    return loss, grad, hess

def penalized_logistic_regression(y, tx, w, lambda_):
    """ Return the penalized logistic loss and its gradient 
        Args:
            y:  shape=(N, 1)
            tx: shape=(N, D)
            w:  shape=(D, 1)
            lambda_: scalar
        Returns:
            loss: scalar number
            gradient: shape=(D, 1) """
            
    pen_log_loss = calculate_loss(y,tx,w) + lambda_*np.linalg.norm(w)**2
    grad = calculate_gradient(y,tx,w)
    pen = (2*lambda_*w).reshape(-1)
    pen_grad = grad + pen
    
    return pen_log_loss, pen_grad
    
def cross_validation_x_log(y, x, k_indices, k, lambda_, degree):
    """ Return the loss of penalized logistic regression with gradient descent
        for a fold corresponding to k_indices 
        The feature x gets expanded by a degree to build the regression matrix tx 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            k_indices: 2D array returned by build_k_indices()
            k: scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
            lambda_: scalar, cf. penalized logistic with GD
            degree: scalar, cf. build_poly()
        Returns:
            loss_tr: scalar(float), rmse = sqrt(2 mse) of the training set
            loss_te: scalar(float), rmse = sqrt(2 mse) of the testing set"""
    
    # get k'th subgroup in test, others in train: TODO
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
    
    # penalized logistic regression with gradient descent
    w_k = logistic_regression_penalized_gradient_descent_demo(y_train, tx_train, lambda_)
    
    # calculate the loss for train and test data
    l_tr = np.exp(-calculate_loss(y_train, tx_train, w_k))
    l_te = np.exp(-calculate_loss(y_test, tx_test, w_k))
    loss_tr = np.sqrt(2*l_tr)
    loss_te = np.sqrt(2*l_te)
    
    return loss_tr, loss_te

def cross_validation_tx_log(y, tx, k_indices, k, lambda_):
    """ Return the loss of penalized logistic regression with gradient descent
        for a fold corresponding to k_indices
        The regression matrix tx is given as an input 
        Args:
            y: shape=(N, ) (N number of events)
            tx: shape=(N, D) (D number of features)
            k_indices: 2D array returned by build_k_indices()
            k: scalar, the k-th fold
            lambda_: scalar, cf. penalized logistic with GD
        Returns:
            loss_tr: scalar(float), rmse = sqrt(2 mse) of the training set
            loss_te: scalar(float), rmse = sqrt(2 mse) of the testing set"""

    # get k'th subgroup in test, others in train: TODO
    k_fold = len(k_indices)
    
    y_test = y[k_indices[k]]
    tx_test = tx[k_indices[k],:]
    
    ind_train = []
    ind_train = np.append(ind_train, k_indices[np.arange(k_fold)!=k])
    ind_train = [int(ind_train[i]) for i in range(len(ind_train))]
                 
    y_train = y[ind_train]
    tx_train = tx[ind_train,:]
    
    # ridge regression
    w_k = logistic_regression_penalized_stochastic_gradient_descent_demo(y_train, tx_train, lambda_)
    
    # calculate the loss for train and test data
    loss_tr = np.sqrt(2*np.exp(-calculate_loss(y_train, tx_train, w_k)))
    loss_te = np.sqrt(2*np.exp(-calculate_loss(y_test, tx_test, w_k)))
    
    return loss_tr, loss_te

def cross_validation_demo_x_log(x, y, degree, k_fold, lambdas):
    """ Return the loss of penalized logistic regression with gradient descent
        for a fold corresponding to k_indices
        The feature x gets expanded by a degree to build the regression matrix tx 
        Args:
            y: shape=(N, ) (N number of events)
            x: shape=(N, )
            degree: integer, degree of the polynomial expansion
            k_fold: integer, the number of folds
            lambdas: shape = (p, ) where p is the number of values of lambda to test
        Returns:
            best_lambda : scalar, value of the best lambda
            best_rmse : scalar, the associated root mean squared error for the best lambda"""
    
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
            loss_tr, loss_te = cross_validation_x_log(y, x, k_indices, k, l, degree)
            loss_tr_sum += loss_tr
            loss_te_sum += loss_te
        rmse_tr = np.append(rmse_tr, loss_tr_sum/k_fold)
        rmse_te = np.append(rmse_te, loss_te_sum/k_fold)
    
    best_ind = np.argmin(rmse_te)
    best_lambda = lambdas[best_ind]
    best_rmse = rmse_te[best_ind]

    # cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    # print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    
    return best_lambda, best_rmse

def cross_validation_demo_tx_log(y, tx, k_fold, lambdas):
    """ Return the loss of penalized logistic regression with gradient descent
        for a fold corresponding to k_indices
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
    
    seed = 1 # set the seed like the one of the best_degree_selection function; put 12 for the graph
    
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    for l in lambdas:
        r_tr = []
        r_te = []
    
        for k in range(k_fold):  # we do this to perform the training using all the data
            r_tr.append(cross_validation_tx_log(y, tx, k_indices, k, l)[0])  
            r_te.append(cross_validation_tx_log(y, tx, k_indices, k, l)[1]) 
        rmse_tr.append(np.mean(r_tr)) # mean of the rmse test for each of the considered lambda
        rmse_te.append(np.mean(r_te))
    
    best_rmse = min(rmse_te)
    minpos = rmse_te.index(best_rmse)  # Get the position of the minimum value of average loss 
    best_lambda = lambdas[minpos]
    #cross_validation_visualization(lambdas, rmse_tr, rmse_te)
    #print("For polynomial expansion up to degree %.f, the choice of lambda which leads to the best test rmse is %.5f with a test rmse of %.3f" % (degree, best_lambda, best_rmse))
    
    return best_lambda, best_rmse, rmse_tr, rmse_te

def best_degree_selection_x_log(x, y, degrees, k_fold, lambdas, seed = 1):
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
        
        best_lambdas.append(cross_validation_demo_x_log(x, y, deg, k_fold, lambdas)[0])
        best_rmses.append(cross_validation_demo_x_log(x, y, deg, k_fold, lambdas)[1])
    
    best_ind = np.argmin(best_rmses)
    best_degree = degrees[best_ind]
    best_lambda = best_lambdas[best_ind]
    best_rmse = best_rmses[best_ind]
    
    return best_degree, best_lambda, best_rmse

