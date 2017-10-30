# -*- coding: utf-8 -*-
import numpy as np
from costs import *



""" Helpers for data preprocessing
        - Divide in 3 datasets
        - Manage invalid values
        - Standardize
"""


def standardize(x, jet = 0):
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


def build_model_data(x):
    #x, mean_x, std_x=standardize(x)
    
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



def put_NaN(data):
    #replace -999 by the mean
    mean_col=np.zeros((data.shape[1],1))

    for d in range(data.shape[1]):
        mean_col[d]=np.mean(data[:,d][data[:,d] !=-999])

        data[:,d][data[:,d] ==-999]=mean_col[d]
        
    return data

    """Replace -999 values by NaN"""
    for ind_feature in range (0, data.shape[1]):
        
        for row in range(0, data.shape[0]):
            if data[row, ind_feature] < -998:
                data[row, ind_feature] = np.nan
    
    return data    


def divide_subset_train(data, y):
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
    
    y_no_jet = y[ind_no_jet]
    y_one_jet = y[ind_one_jet]
    y_more_jet = y[ind_more_jet]
    
    return [[x_no_jet, y_no_jet], [x_one_jet, y_one_jet], [x_more_jet, y_more_jet]]

def divide_subset_test(data):
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
    
    return [[x_no_jet, ind_no_jet], [x_one_jet, ind_one_jet], [x_more_jet, ind_more_jet]]
    


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



def replace_by_mean(x_train, x_test):
    # Replace -999 by the mean
    mean_col=np.zeros((x_train.shape[1],1))

    for d in range(x_train.shape[1]):
        mean_col[d]=np.mean(x_train[:,d][x_train[:,d] !=-999])

        x_train[:,d][x_train[:,d] ==-999]=mean_col[d]
        x_test[:,d][x_test[:,d] ==-999]=mean_col[d]

    return x_train, x_test


def preprocess_datasets(data_train, data_test, y_train):
    
    datasets_train = divide_subset_train(data_train, y_train)
    datasets_test = divide_subset_test(data_test)

    for ind, subset_train in enumerate(datasets_train):
        subset_test = datasets_test[ind]
        # Delete useless columns according to number of jet
        if ind == 0:
            subset_train[0] = np.delete(subset_train[0], [4,5,6,12,22,23,24,25,26,27,28,29], axis = 1)
            subset_test[0] = np.delete(subset_test[0], [4,5,6,12,22,23,24,25,26,27,28,29], axis = 1)
        elif ind == 1:
            subset_train[0] = np.delete(subset_train[0], [4,5,6,12,22,26,27,28,29], axis = 1)
            subset_test[0] = np.delete(subset_test[0], [4,5,6,12,22,26,27,28,29], axis = 1)
        
        # Replace -999 by mean of the corresponding columns
        subset_train[0], subset_test[0] = replace_by_mean(subset_train[0], subset_test[0])
        
        # Standardize
        subset_train[0], _, _ = standardize(subset_train[0])
        subset_test[0], _, _ = standardize(subset_test[0])
        
        # Build model phi a, a^2, a*b, a*c ...
        subset_train[0] = build_model_data(subset_train[0])
        subset_test[0] = build_model_data(subset_test[0])
        
        # Standardize
        subset_train[0], _, _ = standardize(subset_train[0])
        subset_test[0], _, _ = standardize(subset_test[0])
    
        # Normalize in [-1,1]
        #subset_train[0] = normalize_features(subset_train[0])
        #subset_test[0] = normalize_features(subset_test[0])
        print("Features standardized for subset : {}".format(ind))
        
        # Update datasets_train, datasets_test
        datasets_train[ind][0] = subset_train[0]
        datasets_test[ind][0] = subset_test[0]
    
    return datasets_train, datasets_test



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
            

            
"""Helpers for Logistic Regression using Stochastic Gradient Descent"""


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



"""Helpers for Penalized Logistic Regression"""


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient, and hessian."""
    loss = calculate_loss(y,tx,w) + lambda_*np.square(np.linalg.norm(w))
    gradient = calculate_gradient(y,tx,w) + 2*lambda_*w
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """ Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w. """
    loss, gradient = penalized_logistic_regression(y,tx,w,lambda_)
    w = w - gamma*gradient
    return loss, w

    

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