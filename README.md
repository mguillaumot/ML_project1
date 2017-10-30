# ML_project1

In this repository, you can find the implementation of different Machine Learning algorithms to do data analysis on a real world dataset. 

This work takes place in the context of a project in the Machine Learning course of fall semester 2017 at EPFL. It consists in a Kaggle competition based on the Higgs Boson Machine Learning Challenge (2014).

To complete our analysis, we have been working on a dataset made of two csv files : train.csv and test.csv which both gather information on quantities both raw or computed from the raw quantities.
Those dataset need to be in a data folder at the root of the project.

In the scripts folder you will find:

costs.py: 4 cost functions

- compute_mse: compute mean squared error from error (e)
- compute_mae: compute mean absolute error from error (e)
- compute_loss: compute mse or mae from raw data (y,x,w)
- calculate_loss: compute negative log likelihood

helpers.py: Pre-processing functions and helper functions for each regression method to be implemented

- Pre-processing: standardize, de_standardize, sample_data, build_model_data, divide_subset_train, divide_subset_test, replace_by_mean, preprocess_datasets, build_k_indices, compute_accuracy
- Linear regression using gradient descent: compute_gradient
- Linear regression using stochastic gradient descent: compute_stoch_gradient, batch_iter
- Logistic regression using stochastic gradient descent: sigmoid, calculate_gradient, learning_by_gradient_descent
- Penalized logistic regression using stochastic gradient descent: penalized_logistic_regression, learning_by_penalized_gradient
- Cross validation: build_k_indices, compute_accuracy

proj1_helpers.py: 3 functions to load and write csv files needed for analysis of the data and scoring submissions

- load_csv_data: load csv files 
- predict_labels: from computed weights and data outputs the predicted labels
- create_csv_submission: create a csv file with data predictions

plots.py: 2 functions to visualize cross validation results and bias variance decomposition

- cross_validation_visualization
- bias_variance_decomposition_visualizaition

implementations.py: 6 functions describing different regression methods for data analysis

- least_squares_GD: Linear regression using gradient descent
- least_squares_SGD: Linear regression using stochastic gradient descent
- least_squares: Least squares regression using normal equations
- ridge_regression: Ridge regression using normal equations
- logistic_regression: Logistic regression using stochastic gradient descent
- reg_logistic_regression: Regularized logistic regression using stochastic gradient descent

project1.ipynb: python notebook used to test our implementations for this project

run.py: python script used to generate the csv file submitted on Kaggle for final score

In the report folder, you will find our report in a pdf format edited with LaTeX.



