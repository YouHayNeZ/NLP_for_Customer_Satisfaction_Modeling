# Bayesian Ridge Regression (Regression)

# Importing the libraries
from sklearn.linear_model import BayesianRidge
from preprocessing import *
from scipy.stats import uniform
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
from training_helper_func import *

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters
param_dist = {
    'alpha_1': uniform(1e-6, 10),
    'alpha_2': uniform(1e-6, 10),
    'lambda_1': uniform(1e-6, 10),
    'lambda_2': uniform(1e-6, 3)
}

# Hyperparameter tuning & CV results
random_search, results = hpo_and_cv_results(BayesianRidge(), 'outputs/regression/bayesian_ridge/bayesian_ridge_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

# Parallel coordinate plot without max_features and bootstrap
scaler = MinMaxScaler()
results = results.rename(columns={'param_alpha_1': 'alpha_1', 'param_alpha_2': 'alpha_2', 'param_lambda_1': 'lambda_1', 'param_lambda_2': 'lambda_2'})
for param in ['alpha_1', 'alpha_2', 'lambda_1', 'lambda_2']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results = results.drop(columns=['std_test_score', 'rank_test_score'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha=0.25)
plt.legend().remove()
plt.savefig('outputs/regression/bayesian_ridge/bayesian_ridge_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Best model, hyperparameters and predictions
best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                           'outputs/regression/bayesian_ridge/bayesian_ridge_model.pkl', 
                           'outputs/regression/bayesian_ridge/bayesian_ridge_hyperparameters.json', 
                           'outputs/regression/bayesian_ridge/bayesian_ridge_train_preds.csv', 
                           'outputs/regression/bayesian_ridge/bayesian_ridge_test_preds.csv')

# Regression metrics
regression_metrics(y_test, test_preds, 'outputs/regression/bayesian_ridge/bayesian_ridge_scores.json')

# Plot predictions vs real values over time (only regression)
test_preds_vs_real_over_time(test_preds, 'outputs/regression/bayesian_ridge/bayesian_ridge_train_predictions.png')