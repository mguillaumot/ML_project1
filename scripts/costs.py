# -*- coding: utf-8 -*-
import numpy as np



"""Functions used to compute the loss in linear regression using gradient descent and stochastic gradient descent"""


def compute_mse(e):
    return 1/2*np.mean(e**2)


def compute_mae(e):
    return np.mean(np.abs(e))


def compute_loss(y,tx,w):
    e=y-tx.dot(w)
    return compute_mse(e)



"""Functions used to compute the loss in logistic regression using gradient descent"""


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    loss = 0
    for i in range(len(y)):
        loss += np.logaddexp(0,tx[i].dot(w))-y[i]*(tx[i].dot(w))
    return loss