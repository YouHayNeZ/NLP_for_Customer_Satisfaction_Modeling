# K-Nearest Neighbor (Regression)

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import *
from scipy.stats import randint, uniform
import joblib
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters
param_dist = {
    'n_neighbors': randint(1, 150),
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute', 'auto'],
    'leaf_size': randint(1, 200),
    'p': uniform(1, 30),
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=KNeighborsRegressor(), 
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
results.to_csv('outputs/regression/knn/knn_cv_results.csv')

# Parallel coordinate plot
scaler = MinMaxScaler()
results = results.rename(columns={'param_n_neighbors': 'n_neighbors', 'param_weights': 'weights', 'param_algorithm': 'algorithm', 'param_leaf_size': 'leaf_size', 'param_p': 'p', 'param_metric': 'metric'})
for param in ['n_neighbors', 'leaf_size', 'p']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'algorithm', 'weights', 'metric'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
plt.legend().remove()
plt.savefig('outputs/regression/knn/knn_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Count the percentage of each hyperparameter combination
results = results.sort_values(by='mean_test_score', ascending=False)
results_top10 = results.head(int(len(results) * 0.1))
results_bottom10 = results.tail(int(len(results) * 0.1))

# Print length of top 10% and bottom 10% of models
print('Out of top 10% of models (number {}):'.format(len(results_top10)))
for param in ['weights', 'algorithm', 'metric']:
    for value in results_top10[param].unique():
        print(f'{param} = {value}: {len(results_top10[results_top10[param] == value])/len(results_top10)}')

print('Out of bottom 10% of models (number {}):'.format(len(results_bottom10)))
for param in ['weights', 'algorithm', 'metric']:
    for value in results_bottom10[param].unique():
        print(f'{param} = {value}: {len(results_bottom10[results_bottom10[param] == value])/len(results_bottom10)}')

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/regression/knn/knn_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/regression/knn/knn_hyperparameters.json', 'w') as f:
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

train_preds.to_csv('outputs/regression/knn/knn_train_preds.csv')
test_preds.to_csv('outputs/regression/knn/knn_test_preds.csv')

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
with open('outputs/regression/knn/knn_scores.json', 'w') as f:
    json.dump(scores, f)

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/regression/knn/knn_train_predictions.png')
plt.show()