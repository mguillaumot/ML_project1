# -*- coding: utf-8 -*-
import numpy as np
from costs import *



"""Helpers for data pre-processing"""


def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x


def sample_data(y, x, seed, size_samples):
    """sample from dataset."""
    np.random.seed(seed)
    num_observations = y.shape[0]
    random_permuted_indices = np.random.permutation(num_observations)
    y = y[random_permuted_indices]
    x = x[random_permuted_indices]
    return y[:size_samples], x[:size_samples]


def build_model_data(height, weight):
    """Form (y,tX) to get regression data in matrix form."""
    y = weight
    x = height
    num_samples = len(y)
    tx = np.c_[np.ones(num_samples), x]
    return y, tx



"""Helpers for Gradient Descent"""


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    err=y-tx.dot(w)
    grad=-tx.T.dot(err)/len(err)  
    return grad,err



"""Helpers for Stochastic Gradient Descent"""


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    err=y-tx.dot(w)
    grad=-tx.T.dot(err)/len(err)
    return grad,err


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
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
            

            
"""Helpers for Logistic Regression using Gradient Descent"""


def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(-np.logaddexp(0, -t))


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



"""Helpers for Logistic Regression using Newton's method"""


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



"""Helpers for Penalized Logistic Regression"""


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