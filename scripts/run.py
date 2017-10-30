# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementation import *

from costs import compute_mse, compute_loss
from proj1_helpers import *
from helpers import *

def run():
    """Build our model on the train dat"""
    #Load the train Data
    DATA_TRAIN_PATH = '../../train.csv'
    #DATA_TRAIN_PATH = '../train.csv'
    y, X, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)

    #Clean and Standardize the data
    tx=clean_and_std_data(X)

    #change the -1 from y to 0
    y[y ==-1]=0

    #Build our model phi=[1 X X^2 sqrt(abs(X)) abs(x)]
    phi=build_model(tx,4)

    # Set max_iter w_initial and gamma for logistic regression
    max_iter=10000
    gamma=0.00001

    w_initial=np.zeros((phi.shape[1],1))

    loss_w,weights=logistic_regression(y, phi, w_initial, max_iter,gamma)


    """Generate predictions and save ouput in csv format for submission:"""
    #Load the testData
    DATA_TEST_PATH = '../../test.csv'
    _, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

    #Clean and Standardize the data
    data_test=clean_and_std_data(tX_test)

    #Build our model phi=[1 X X^2 sqrt(abs(X)) abs(X)]
    phi_test=build_model(data_test,4)

    #make our prediction
    y_pred= predict_labels_sigmoid(weights,phi_test)

    #change the 0 by 1 for the submission
    y_pred[y_pred ==0]=-1

    #create cvs submission for ou prediction called prediction_43.csv
    OUTPUT_PATH = 'prediction.csv'
    create_csv_submission(ids_test, y_pred, OUTPUT_PATH)

def preprocessing(data, y):
    data = put_NaN(data)
    datasets = divide_subset(data, y)
    
    # Preprocess datasets_train
    for ind, subset in enumerate(datasets):
        print('Train, before preprocessing : {} '.format(subset[0].shape))
    if ind == 0:
        subset[0] = preprocess_data_train(subset_train[0], jet = 0)
    elif ind == 1:
        subset_train[0] = preprocess_data_train(subset_train[0], jet = 1)
    elif ind == 2:
        subset_train[0] = preprocess_data_train(subset_train[0], jet = 23)

    print('Train, after preprocessing : {} '.format(subset_train[0].shape))
    datasets_train[ind][0] = subset_train[0]
        
