# Bayesian Ridge Regression (Regression)

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import *
from scipy.stats import uniform
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters
param_dist = {
    'alpha_1': uniform(1e-6, 10),
    'alpha_2': uniform(1e-6, 10),
    'lambda_1': uniform(1e-6, 10),
    'lambda_2': uniform(1e-6, 3)
}

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=BayesianRidge(), 
                                   param_distributions=param_dist, 
                                   n_iter=1500, 
                                   cv=5, 
                                   verbose=2, 
                                   scoring='neg_mean_absolute_error',
                                   random_state=42, 
                                   n_jobs=-1)
random_search.fit(X_train, y_train)

# Save results of RandomizedSearchCV
results = pd.DataFrame(random_search.cv_results_)
interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
results = results[interested_columns]
results = results.sort_values(by='rank_test_score')
results['mean_test_score'] = results['mean_test_score']
results.to_csv('outputs/regression/bayesian_ridge/bayesian_ridge_cv_results.csv')

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

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/regression/bayesian_ridge/bayesian_ridge_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/regression/bayesian_ridge/bayesian_ridge_hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)

# Make predictions, save them as data frame and set flown date as index
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Correct classes again: add +1 to predictions & real values to get the real rating
train_preds = np.round(train_preds + 1)
test_preds = np.round(test_preds + 1)
y_train = y_train + 1
y_test = y_test + 1

train_preds = pd.DataFrame({'Predicted Overall Rating': train_preds, 'Real Overall Rating': y_train, 'Date Flown': datetime_train['Date Flown']}).set_index('Date Flown')
test_preds = pd.DataFrame({'Predicted Overall Rating': test_preds, 'Real Overall Rating': y_test, 'Date Flown': datetime_test['Date Flown']}).set_index('Date Flown')

train_preds.to_csv('outputs/regression/bayesian_ridge/bayesian_ridge_train_preds.csv')
test_preds.to_csv('outputs/regression/bayesian_ridge/bayesian_ridge_test_preds.csv')

# Compute metrics
mae = mean_absolute_error(y_test, test_preds['Predicted Overall Rating'])
mse = mean_squared_error(y_test, test_preds['Predicted Overall Rating'])
r2 = r2_score(y_test, test_preds['Predicted Overall Rating'])

# Save scores
scores = {
    'mae': mae,
    'mse': mse,
    'r2': r2
}
print(scores)
with open('outputs/regression/bayesian_ridge/bayesian_ridge_scores.json', 'w') as f:
    json.dump(scores, f)

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/regression/bayesian_ridge/bayesian_ridge_predictions.png')
plt.show()