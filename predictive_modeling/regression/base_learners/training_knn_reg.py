# K-Nearest Neighbor (Regression)

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

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv', classification=False)

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

if __name__ == '__main__':
    main()