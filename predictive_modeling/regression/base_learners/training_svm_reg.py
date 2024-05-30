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
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

    # Define the range of hyperparameters
    param_dist = {
        'C': uniform(0.000001, 100),
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': ['scale', 'auto'],
        'degree': randint(1, 10),
        'epsilon': uniform(0.000001, 1),
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(SVR(), 'outputs/regression/svm/svm_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_C': 'C', 'param_kernel': 'kernel', 'param_gamma': 'gamma', 'param_degree': 'degree', 'param_epsilon': 'epsilon'})
    for param in ['C', 'degree']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'kernel', 'gamma'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/regression/svm/svm_parallel_coordinates.png')
    plt.show()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['kernel', 'gamma', 'degree'])

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/regression/svm/svm_model.pkl', 
                            'outputs/regression/svm/svm_hyperparameters.json', 
                            'outputs/regression/svm/svm_train_preds.csv', 
                            'outputs/regression/svm/svm_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/regression/svm/svm_scores.json')

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/regression/svm/svm_train_predictions.png')

if __name__ == '__main__':
    main()