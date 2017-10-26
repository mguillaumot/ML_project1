# -*- coding: utf-8 -*-
import numpy as np

"""Function used to compute the loss."""

def compute_mse(e):
    return 1/2*np.mean(e**2)

def compute_mae(e):
    return np.mean(np.abs(e))

def compute_loss(y,tx,w):
    e=y-tx.dot(w)
    return compute_mse(e)

"""Build polynomial"""

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly=np.ones((len(x),1))
    for deg in range(1,degree+1):
        poly=np.c_[poly, np.power(x,deg)]
    return poly

"""Gradient Descent"""

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err=y-tx.dot(w)
    grad=-tx.T.dot(err)/len(err)  
    return grad,err

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # compute loss, gradient
        grad,err=compute_gradient(y,tx,w)
        loss=compute_mse(err)
        w=w-gamma*grad
        # store w and loss
        # ws.append(w)
        # losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return w, loss
    


"""Stochastic Gradient Descent"""

from proj1_helpers import batch_iter

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err=y-tx.dot(w)
    grad=-tx.T.dot(err)/len(err)
    return grad,err

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    batch_size=1
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            grad,_=compute_stoch_gradient(y_batch,tx_batch,w)
            w=w-gamma*grad
            loss=compute_loss(y,tx,w)
            # store w and loss
            # ws.append(w)
            # losses.append(loss)
            
        # print("SGD({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
             # bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return w, loss



"""
Least Squares
"""

def least_squares(y, tx):
    """calculate the least squares solution."""
    a=tx.T.dot(tx)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    mse=compute_loss(y, tx, w)
    loss=np.sqrt(2*mse)
    return w, loss



"""
Ridge Regression
"""

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    aI=2*tx.shape[0]*lambda_*np.identity(tx.shape[1])
    a=tx.T.dot(tx)+aI
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    mse=compute_loss(y, tx, w)
    loss=np.sqrt(2*mse)
    return w, loss



""" Logistic regression using gradient descent """

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(-np.logaddexp(0, -t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0
    for i in range(len(y)):
        loss += np.logaddexp(0,tx[i].dot(w))-y[i]*(tx[i].dot(w))
    return loss

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx, w)) - y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """ Do one step of gradient descen using logistic regression.
    Return the loss and the updated w. """
    loss = calculate_loss(y,tx,w)
    grad = calculate_gradient(y,tx,w)
    w = w - gamma*grad
    return loss, w

def logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    w = initial_w
    y = (y+1)/2
    losses = []

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        losses.append(loss)
        
        # converge criteria
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w, loss


    
    
""" Logistic regression using Newton's method """

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    pred = sigmoid(tx.dot(w))
    pred = np.diag(pred.T[0])
    S = np.multiply(pred,(1-pred))
    return tx.T.dot(S).dot(tx)

def logistic_regression_calculation(y, tx, w):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y,tx,w)
    gradient = calculate_gradient(y,tx,w)
    hessian = calculate_hessian(y,tx,w)
    return loss, gradient, hessian

def learning_by_newton_method(y, tx, w):
    """ Do one step on Newton's method.
    return the loss and updated w. """
    loss, gradient, hessian = logistic_regression_calculation(y,tx,w)
    w = w - np.linalg.solve(hessian,gradient)
    return loss, w

def logistic_regression_newton_method(y, tx, lambda_, initial_w, max_iters, gamma):
    # init parameters
    threshold = 1e-8
    w = initial_w
    y = (y+1)/2
    losses = []

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w)
        losses.append(loss)
            
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return w, loss



""" Regularized logistic regression using gradient descent """

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y,tx,w) + lambda_*np.square(np.linalg.norm(w))
    gradient = calculate_gradient(y,tx,w) + 2*lambda_*w
    hessian = calculate_hessian(y,tx,w)
    return loss, gradient

def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """ Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w. """
    loss, gradient = penalized_logistic_regression(y,tx,w,lambda_)
    w = w - gamma*gradient
    return loss, w

def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w
    y = (y+1)/2

    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        
        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
            
    return w, loss




"""
Cross Validation
"""

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation(y, x, w, k_indices, k, lambda_):
    """return the loss of ridge regression."""
    te_indices=k_indices[k]
    tr_indices=k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_indices=tr_indices.reshape(-1)
    
    tx_te=x[te_indices]
    y_te=y[te_indices]
    tx_tr=x[tr_indices]
    y_tr=y[tr_indices]
    
    loss_tr=np.sqrt(2*compute_loss(y_tr, tx_tr, w))
    loss_te=np.sqrt(2*compute_loss(y_te, tx_te, w))
    return loss_tr, loss_te