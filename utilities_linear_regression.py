import numpy as np

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """ Generate a minibatch iterator for a dataset
        Data can be randomly shuffled to avoid ordering in the original data 
        messing with the randomness of the minibatches """
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
    """ Calculate the loss using either MSE or MAE """
    return compute_mse(y, tx, w)

def compute_gradient(y, tx, w):
    """ Computes the gradient at w """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def compute_stoch_gradient(y, tx, w):
    """ Compute a stochastic gradient """
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """ The Gradient Descent (GD) algorithm """
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
        
    return losses, ws

def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """ The Stochastic Gradient Descent algorithm (SGD) """
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
    
    return losses, ws

def build_poly(x, degree):
    """ Polynomial basis functions for input data x, for j=0 up to j=degree """
    phi_x = np.zeros((x.size, degree+1))
    for i in range(len(x)):
        curr_row = np.array([x[i]**deg for deg in range(degree+1)])
        phi_x[i] = curr_row
        
    return phi_x

def least_squares(y, tx):
    """ The Least Squares algorithm (LS) """
    w = np.linalg.solve(np.dot(tx.T,tx), np.dot(tx.T,y))
    mse = compute_mse(y, tx, w)
    
    return w, mse

def ridge_regression(y, tx, lambda_):
    """ The Ridge Regression algorithm (RR) """
    N = y.size
    D = tx[0,:].size
    I = np.identity(D)
    A = np.dot(tx.T,tx)+(lambda_*2*N)*I
    b = np.dot(tx.T,y)
    w = np.linalg.solve(A, b)
    
    loss = compute_loss(y, tx, w)
    
    return w, loss

def build_k_indices(y, k_fold, seed):
    """ Build k indices for k-fold """
    num_row = y.shape[0]
    interval = int(num_row / k_fold) # Here it computes the number of intervals
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    
    return np.array(k_indices)

def cross_validation_x_lin(y, x, k_indices, k, lambda_, degree):
    """ Return the loss of ridge regression for a fold corresponding to k_indices 
        The feature x gets expanded by a degree to build the regression matrix tx """
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
        The regression matrix tx is given as an input """

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
        The feature x gets expanded by a degree to build the regression matrix tx """
    
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
        The regression matrix tx is given as an input """
    
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
    
    return best_lambda, best_rmse

def best_degree_selection_x_lin(x, y, degrees, k_fold, lambdas, seed = 1):
    """ Cross validation over regularisation parameter lambda and degree """
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
