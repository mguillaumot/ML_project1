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

    print("Data loaded")
    
    # Preprocess data
    datasets_train, datasets_test = preprocess_datasets(X_train, X_test, y_train = y)
    
    print("Preprocessing done")
    
    # **** Parameters **** 
    # Set max_iter w_initial and gamma for logistic regression
    max_iters = 100
    gamma = 0.6
    lambdas = [0.0013894954943731374, 2.2758459260747865e-05, 1e-05]

    y_pred_final = np.zeros((len(ids_test),1))
    
    # For each case of jet number
    for ind, subset_train in enumerate(datasets_train):
        print("Starting subset " + str(ind))
        
        w, loss = ridge_regression(y = subset_train[1], tx = subset_train[0], lambda_ = lambdas[ind])
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