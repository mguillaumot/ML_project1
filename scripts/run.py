# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementations import *

from costs import compute_mse, compute_loss
from proj1_helpers import *
from helpers import *

def run():
    """Build our model on the train dat"""
    #Load the train Data
    DATA_TRAIN_PATH = '../../data/train.csv'
    #DATA_TRAIN_PATH = '../train.csv'
    y, X, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

    #Clean and Standardize the data
    tx=clean_and_std_data(X)
    
    
    #Build our model phi a, a^2, a*b, a*c ...
    phi=build_model(tx)

    # Set max_iter w_initial and gamma for logistic regression
    max_iter=100
    gamma=0.5
    #lambdas = [0.00005,0.000001389495494373136,0.00000007196856730011513]
    lambda_ = 0.00005
    
    #w_initial=np.zeros((phi.shape[1],1))
    
    #w, loss = method(y = y_train, tx = poly_train, lambda_ = lambda_)
    w, loss = ridge_regression(y, phi, lambda_)


    """Generate predictions and save ouput in csv format for submission:"""
    #Load the testData
    DATA_TEST_PATH = '../../data/test.csv'
    _, X_test, ids_test = load_csv_data(DATA_TEST_PATH, sub_sample=False)

    #Clean and Standardize the data
    tx_test=clean_and_std_data(X_test)

    #Build our model phi_test a, a^2, a*b, a*c ...
    phi_test=build_model(tx_test)

    #make our prediction
    y_pred= predict_labels(w,phi_test)


    #create cvs submission for ou prediction called prediction_43.csv
    OUTPUT_PATH = 'prediction.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)
    print("prediction ready");

def clean_and_std_data(X):
    #replace -999 by the mean
    mean_col=np.zeros((X.shape[1],1))

    for d in range(X.shape[1]):
        mean_col[d]=np.mean(X[:,d][X[:,d] !=-999])

        X[:,d][X[:,d] ==-999]=mean_col[d]


    # standardize the data
    centered_data = X - np.mean(X, axis=0)

    std_data = centered_data / np.std(centered_data, axis=0)

     #put to zero if the data <10**(-10)
    return std_data

    X= put_NaN(X)
    tx = divide_subset(X, y)
    tx, mean_x, std =standardize(tx)
    """
    for ind, subset_train in enumerate(datasets_train):
        if ind == 0:
            subset_train[0] = preprocess_data_train(subset_train[0], jet = 0)
        elif ind == 1:
            subset_train[0] = preprocess_data_train(subset_train[0], jet = 1)
        elif ind == 2:
            subset_train[0] = preprocess_data_train(subset_train[0], jet = 23)
        datasets_train[ind][0] = subset_train[0]
    
    """
    return tx
run()