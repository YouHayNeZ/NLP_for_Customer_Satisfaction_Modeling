# Multi-Layer Perceptron (Regression)

# Importing the libraries
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import loguniform, randint
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

    # Define the range of hyperparameters
    param_dist = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': loguniform(0.0001, 0.1),
        'learning_rate': ['constant','adaptive'],
        'max_iter': randint(50, 1000),
        'early_stopping': [True]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(MLPRegressor(), 'outputs/regression/mlp/mlp_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error', n_iter=50)

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_hidden_layer_sizes': 'hidden_layer_sizes', 'param_activation': 'activation', 'param_solver': 'solver', 'param_alpha': 'alpha', 'param_learning_rate': 'learning_rate', 'param_max_iter': 'max_iter', 'param_early_stopping': 'early_stopping'})
    for param in ['alpha', 'max_iter']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'hidden_layer_sizes', 'activation', 'solver', 'learning_rate', 'early_stopping'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/regression/mlp/mlp_parallel_coordinates.png')
    plt.show()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate'])

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/regression/mlp/mlp_model.pkl', 
                            'outputs/regression/mlp/mlp_hyperparameters.json', 
                            'outputs/regression/mlp/mlp_train_preds.csv', 
                            'outputs/regression/mlp/mlp_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/regression/mlp/mlp_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/regression/mlp/mlp_train_predictions.png')

if __name__ == '__main__':
    main()