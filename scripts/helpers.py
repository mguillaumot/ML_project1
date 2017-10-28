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

def build_model(x):
    x, mean_x, std_x=standardize(x)
    
    n = int((x.shape[1]+2)*(x.shape[1]+1)/2)
    phi = np.zeros((x.shape[0],n))
    for j in range(x.shape[0]):
        features = np.concatenate(([1],x[j,:]))
        m = len(features)
        matrix = np.outer(features, features)
        np.tril(np.outer(features, features)).reshape(-1)
        iu1 = np.triu_indices(m)
        phi[j,:] = matrix[iu1]
        
    # x, mean_x, std_x=standardize(phi)
        
    return phi    


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


""" Helpers for data preprocessing
        - Divide in 3 datasets
        - Manage invalid values
        - Standardize
"""

def put_NaN(data):
    """Replace -999 values by NaN"""
    for ind_feature in range (0, data.shape[1]):
        
        for row in range(0, data.shape[0]):
            if data[row, ind_feature] < -998:
                data[row, ind_feature] = np.nan
    
    return data    

def divide_subset(data, y = None):
    """Divides initial dataset into 3 sub-dataset"""
    # *** Case no jet ***
    ind_no_jet = np.where(data[:,22] == 0)[0]
    x_no_jet = data[ind_no_jet]
    
    # *** Case one jet ***
    ind_one_jet = np.where(data[:,22] == 1)[0]
    x_one_jet = data[ind_one_jet]
    
    # *** Case two or more jet ***
    ind_more_jet = np.where(data[:,22] > 1)[0]
    x_more_jet = data[ind_more_jet]
    
    if y == None:
        return [[x_no_jet, ind_no_jet], [x_one_jet, ind_one_jet], [x_more_jet, ind_more_jet]]
    
    y_no_jet = y[ind_no_jet]
    y_one_jet = y[ind_one_jet]
    y_more_jet = y[ind_more_jet]
    
    return [[x_no_jet, y_no_jet], [x_one_jet, y_one_jet], [x_more_jet, y_more_jet]]


def remove_NaN_columns(data):
    """Remove columns full of NaN"""
    # Get indices of column full of NaN
    ind_col_nan = np.where(~np.isnan(data).all(axis=0))
    
    # Keep only others columns
    data = data[:,ind_col_nan[0]]
    
    return data

def replace_NaN_by_mean(data):
    """Replace persistent NaN by mean of the feature"""
    # If NaN value, it is replace by the mean of the feature 
    ind_col_nan = np.where(np.isnan(data).any(axis=0))

    # For each column where a NaN appears
    for col in ind_col_nan[0]:
        # Get the indices of the row where the NaN appears in the current column
        ind_row_nan = []
        for i in range(data[:,col].shape[0]):
            if np.isnan(data[i,col]) :
                ind_row_nan.append(i)

        # Compute mean of the column, excluding the nan 
        x_row_curr = np.delete(data[:,col], ind_row_nan)
        mean_no_nan = round(np.mean(x_row_curr),3)

        # Replace NaN value of the row by the mean value
        for i in range(len(ind_row_nan)):
            data[ind_row_nan[i],col] = mean_no_nan
        
    return data

def replace_NaN_by_mean_test(data_test, data_train):
    """Replace persistent NaN by mean of the feature for the test_dataset"""
    # If NaN value, it is replace by the mean of the feature 
    ind_col_nan = np.where(np.isnan(data_test).any(axis=0))

    # For each column where a NaN appears
    for col in ind_col_nan[0]:
        # Get the indices of the row where the NaN appears in the current column
        ind_row_nan = []
        for i in range(data_test[:,col].shape[0]):
            if np.isnan(data_test[i,col]) :
                data_test[i,col] = np.mean(data_train[:,col])
        
    return data_test

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis = 0)
    x = x - mean_x
    std_x = np.std(x, axis = 0)
    x = x / std_x
    
    return x, mean_x, std_x

def normalize_features(data):
    """ Normalization of each feature between -1 and 1"""
    for ind_feature in range (0, data.shape[1]):
        ft_max = np.max( data[:, ind_feature], axis= 0)
        ft_min = np.min( data[:, ind_feature], axis = 0)
        
        for row in range(0, data.shape[0]):
            ft_crt = data[row, ind_feature] 
            ft_norm = 2 * ( ft_crt - ft_min) / (ft_max - ft_min) - 1           
            data[row, ind_feature] = ft_norm
    
    return data    

def preprocess_data_train(data_train, jet = 0):
    # Remove unused columns, full of NaN
    data_train = remove_NaN_columns(data_train)
    
    # Remove features PRI_jet_num & PRI_jet_all_pt
    if (jet == 0 or jet == 1):
        data_train = np.delete(data_train, -1, axis = 1)
        data_train = np.delete(data_train, -1, axis = 1)
        
    # TODO : 1 more row to manage for jet = 1
    
    # Replace NaN by mean of the feature
    data_train = replace_NaN_by_mean(data_train)
    
    # Standardize
    data_train, mean_data, std_data = standardize(data_train)
        
    # Normalize in [-1,1]
#     data_train = normalize_features(data_train)
    
    return data_train

def preprocess_data_test(data_test, data_train, jet = 0):
    # Remove unused columns, full of NaN
    data_test = remove_NaN_columns(data_test)
    
    # Remove features PRI_jet_num & PRI_jet_all_pt
    if (jet == 0 or jet == 1):
        data_test = np.delete(data_test, -1, axis = 1)
        data_test = np.delete(data_test, -1, axis = 1)
        
    # TODO : 1 more row to manage for jet = 1
    
    # Replace NaN by mean of the feature
    data_test = replace_NaN_by_mean_test(data_test, data_train)
    
    # Standardize
    data_test, _, _ = standardize(data_test)
    
    # Normalize in [-1,1]
#     data_test = normalize_features(data_test)
    
    return data_test


"""Helpers cross validation"""

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)


def compute_accuracy(y_pred, y):
    """Compute accuracy"""
    n_valid = 0
    for ind in range(0, len(y)):
        if y[ind] == y_pred[ind]:
            n_valid += 1
    accuracy = n_valid / len(y)
    return accuracy