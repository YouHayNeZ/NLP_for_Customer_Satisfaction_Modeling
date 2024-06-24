# Bayesian Ridge Regression (Regression)

# Importing the libraries
from sklearn.linear_model import BayesianRidge
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

# K-Nearest Neighbors (Regression)

# Importing the libraries
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

# Multi-Layer Perceptron (Regression)

# Importing the libraries
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import uniform, randint
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

# Random Forest (Regression)

# Importing the libraries
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

# Support Vector Machine (Regression)

# Importing the libraries
from sklearn.svm import SVR
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *



def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    param_dist = {
        'alpha_1': uniform(1e-6, 10),
        'alpha_2': uniform(1e-6, 10),
        'lambda_1': uniform(1e-6, 10),
        'lambda_2': uniform(1e-6, 3),
        'tol': [0.0001, 0.005]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(BayesianRidge(), 'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_alpha_1': 'alpha_1', 'param_alpha_2': 'alpha_2', 'param_lambda_1': 'lambda_1', 'param_lambda_2': 'lambda_2', 'param_tol': 'tol'})
    for param in ['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'tol']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results = results.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_model.pkl', 
                            'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_hyperparameters.json', 
                            'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_train_preds.csv', 
                            'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_train_predictions.png')






    # K-Nearest Neighbor (Regression)


    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    param_dist = {
        'n_neighbors': randint(5, 500),
        'weights': ['uniform', 'distance'],
        'algorithm': ['ball_tree', 'kd_tree', 'brute'],
        'leaf_size': randint(1, 500),
        'p': uniform(1, 100),
        'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(KNeighborsRegressor(), 'outputs/predictive_modeling/regression/base_learners/knn/knn_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_neighbors': 'n_neighbors', 'param_weights': 'weights', 'param_algorithm': 'algorithm', 'param_leaf_size': 'leaf_size', 'param_p': 'p', 'param_metric': 'metric'})
    for param in ['n_neighbors', 'leaf_size', 'p']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'algorithm', 'weights', 'metric'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/base_learners/knn/knn_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['weights', 'algorithm', 'metric'])

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/regression/base_learners/knn/knn_model.pkl', 
                            'outputs/predictive_modeling/regression/base_learners/knn/knn_hyperparameters.json', 
                            'outputs/predictive_modeling/regression/base_learners/knn/knn_train_preds.csv', 
                            'outputs/predictive_modeling/regression/base_learners/knn/knn_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/base_learners/knn/knn_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/base_learners/knn/knn_train_predictions.png')
    


    # MLP

    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of layer sizes
    layer_sizes = [25, 50, 75, 100,125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500]

    # Function to generate hidden layer sizes with random number of layers between 2 and 20
    def random_hidden_layers():
        num_layers = random.randint(3, 7)
        return tuple(random.choice(layer_sizes) for _ in range(num_layers))

    # Define the range of hyperparameters
    param_dist = {
        'hidden_layer_sizes': [random_hidden_layers() for _ in range(100)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': uniform(0.0001, 1.0),
        'learning_rate': ['constant','adaptive', 'invscaling'],
        'max_iter': randint(50, 5000),
        'early_stopping': [True],
        'n_iter_no_change': randint(1, 10),
        'tol': uniform(0.0001, 0.005)
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(MLPRegressor(), 'outputs/predictive_modeling/regression/base_learners/mlp/mlp_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error', n_iter=50)

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_hidden_layer_sizes': 'hidden_layer_sizes', 'param_activation': 'activation', 'param_solver': 'solver', 'param_alpha': 'alpha', 'param_learning_rate': 'learning_rate', 'param_max_iter': 'max_iter', 'param_early_stopping': 'early_stopping', 'param_n_iter_no_change': 'n_iter_no_change', 'param_tol': 'tol'})
    for param in ['alpha', 'max_iter', 'n_iter_no_change', 'tol']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'hidden_layer_sizes', 'activation', 'solver', 'learning_rate', 'early_stopping'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/base_learners/mlp/mlp_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate'])

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/regression/base_learners/mlp/mlp_model.pkl', 
                            'outputs/predictive_modeling/regression/base_learners/mlp/mlp_hyperparameters.json', 
                            'outputs/predictive_modeling/regression/base_learners/mlp/mlp_train_preds.csv', 
                            'outputs/predictive_modeling/regression/base_learners/mlp/mlp_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/base_learners/mlp/mlp_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/base_learners/mlp/mlp_train_predictions.png')




    # Random Forest (Regression)

    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    param_dist = {
        'n_estimators': randint(10, 2000),
        'max_features': ["sqrt", "log2", None],
        'max_depth': randint(1, 200),
        'min_samples_split': randint(2, 40),
        'min_samples_leaf': randint(1, 25),
        'bootstrap': [True, False]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(RandomForestRegressor(criterion='absolute_error'), 'outputs/predictive_modeling/regression/base_learners/rf/rf_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_max_features': 'max_features', 'param_max_depth': 'max_depth', 'param_min_samples_split': 'min_samples_split', 'param_min_samples_leaf': 'min_samples_leaf', 'param_bootstrap': 'bootstrap'})
    for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['max_features', 'bootstrap', 'std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/base_learners/rf/rf_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['max_features', 'bootstrap'])

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/regression/base_learners/rf/rf_model.pkl', 
                            'outputs/predictive_modeling/regression/base_learners/rf/rf_hyperparameters.json', 
                            'outputs/predictive_modeling/regression/base_learners/rf/rf_train_preds.csv', 
                            'outputs/predictive_modeling/regression/base_learners/rf/rf_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/base_learners/rf/rf_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/base_learners/rf/rf_train_predictions.png')

    # Create feature importance plot for top 15
    feature_importance = best_model.feature_importances_
    features = X_train.columns
    feature_importance_scores = dict(zip(features, feature_importance))
    with open('outputs/predictive_modeling/regression/base_learners/rf/rf_feature_importance.json', 'w') as f:
        json.dump(feature_importance_scores, f)

    feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.savefig('outputs/predictive_modeling/regression/base_learners/rf/rf_feature_importance.png')
    plt.close()




    # Support Vector Machine (Regression)

    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    param_dist = {
        'C': uniform(0.000001, 100),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': randint(1, 15),
        'epsilon': uniform(0.000001, 1),
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(SVR(), 'outputs/predictive_modeling/regression/base_learners/svm/svm_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_C': 'C', 'param_kernel': 'kernel', 'param_gamma': 'gamma', 'param_degree': 'degree', 'param_epsilon': 'epsilon'})
    for param in ['C', 'degree']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'kernel', 'gamma'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/base_learners/svm/svm_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['kernel', 'gamma', 'degree'])

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/regression/base_learners/svm/svm_model.pkl', 
                            'outputs/predictive_modeling/regression/base_learners/svm/svm_hyperparameters.json', 
                            'outputs/predictive_modeling/regression/base_learners/svm/svm_train_preds.csv', 
                            'outputs/predictive_modeling/regression/base_learners/svm/svm_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/base_learners/svm/svm_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/base_learners/svm/svm_train_predictions.png')

if __name__ == '__main__':
    main()


