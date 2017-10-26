# -*- coding: utf-8 -*-
import numpy as np
from helpers import *
from costs import *



"""Linear regression using Gradient Descent"""


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
    
    

"""Linear regression using Stochastic Gradient Descent"""


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



"""Least Squares"""


def least_squares(y, tx):
    """calculate the least squares solution."""
    a=tx.T.dot(tx)
    b=tx.T.dot(y)
    w=np.linalg.solve(a,b)
    mse=compute_loss(y, tx, w)
    loss=np.sqrt(2*mse)
    return w, loss



"""Ridge Regression"""


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