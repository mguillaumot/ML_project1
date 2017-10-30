# -*- coding: utf-8 -*-
"""some helper functions for project 1."""
import csv
import numpy as np
from implementations import *

from costs import compute_mse, compute_loss
from proj1_helpers import *
from helpers import *

def run():
    """Build our model on the train data"""
    # Load the train and test datatets
    DATA_TRAIN_PATH = '../data/train.csv'
    DATA_TEST_PATH = '../data/test.csv'
    
    y, X_train, ids = load_csv_data(DATA_TRAIN_PATH, sub_sample=False)
    _, X_test, ids_test = load_csv_data(DATA_TEST_PATH, sub_sample=False)

    # Preprocess data
    datasets_train, datasets_test = preprocess_datasets(X_train, X_test)
    
    
    # **** Parameters **** 
    # Set max_iter w_initial and gamma for logistic regression
    max_iter=100
    gamma=0.5
    #lambdas = [0.00005,0.000001389495494373136,0.00000007196856730011513]
    lambda_ = 0.00005
    
    y_pred_final = np.zeros((len(ids_test),1))
    
    # For each case of jet number
    for ind, subset_train in enumerate(datasets_train):
        
        #w_initial=np.zeros((phi.shape[1],1))
        #w, loss = method(y = y_train, tx = poly_train, lambda_ = lambda_)
        w, loss = ridge_regression(y = subset_train[1], tx = subset_train[0], lambda_ = lambda_)


        """Generate predictions and save ouput in csv format for submission:"""
        
        
        # Prediction for test dataset
        subset_test = datasets_test[ind][0]
        y_pred = predict_labels(w, subset_test)   
        
        np.put(y_pred_final, datasets_test[ind][1], y_pred)

    #create cvs submission for ou prediction called prediction_43.csv
    OUTPUT_PATH = 'predictions.csv'
    create_csv_submission(ids_test, y_pred_final, OUTPUT_PATH)
    
    print("Prediction ready in " + str(OUTPUT_PATH));
    
    
if __name__ == '__main__':
    run()