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

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv', classification=False)

    # Define the range of layer sizes
    layer_sizes = [50, 75, 100, 150, 175, 200, 250, 275, 300, 350, 375, 400, 425, 450, 475, 500]

    # Function to generate hidden layer sizes with random number of layers between 2 and 20
    def random_hidden_layers():
        num_layers = random.randint(4, 6)
        return tuple(random.choice(layer_sizes) for _ in range(num_layers))

    # Define the range of hyperparameters
    # param_dist = {
    #     'hidden_layer_sizes': [random_hidden_layers() for _ in range(100)],
    #     'activation': ['relu'],
    #     'solver': ['adam', 'sgd'],
    #     'alpha': uniform(0.0001, 1.0),
    #     'learning_rate': ['constant','adaptive', 'invscaling'],
    #     'max_iter': [7500],
    #     'early_stopping': [True],
    #     'n_iter_no_change': randint(10, 15),
    #     'tol': uniform(0.0001, 0.005)
    # }

    param_dist = {
        'hidden_layer_sizes': [(450, 425, 350, 400)],
        'activation': ['relu'],
        'solver': ['adam'],
        'alpha': [0.28103450968738075],
        'learning_rate': ['constant'],
        'max_iter': [7500],
        'early_stopping': [True],
        'n_iter_no_change': [14],
        'tol': [0.00017818203370596967]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(MLPRegressor(), 'outputs/predictive_modeling/regression/base_learners/mlp/mlp_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error', n_iter=1, cv=10)

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_hidden_layer_sizes': 'hidden_layer_sizes', 'param_activation': 'activation', 'param_solver': 'solver', 'param_alpha': 'alpha', 'param_learning_rate': 'learning_rate', 'param_max_iter': 'max_iter', 'param_early_stopping': 'early_stopping', 'param_n_iter_no_change': 'n_iter_no_change', 'param_tol': 'tol'})
    results = results.dropna(subset=['mean_test_score'])
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

if __name__ == '__main__':
    main()