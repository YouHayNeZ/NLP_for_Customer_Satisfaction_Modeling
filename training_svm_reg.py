# Support Vector Machine (Regression)

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import *
from scipy.stats import randint, uniform
import joblib
import matplotlib.pyplot as plt
import json
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates

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

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=SVR(),
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
results.to_csv('outputs/regression/svm/svm_cv_results.csv')

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

# Count the percentage of each hyperparameter combination
results = results.sort_values(by='mean_test_score', ascending=False)
results_top10 = results.head(int(len(results) * 0.1))
results_bottom10 = results.tail(int(len(results) * 0.1))

# Print length of top 10% and bottom 10% of models
print('Out of top 10% of models (number {}):'.format(len(results_top10)))
for param in ['kernel', 'gamma']:
    for value in results_top10[param].unique():
        print(f'{param} = {value}: {len(results_top10[results_top10[param] == value])/len(results_top10)}')

print('Out of bottom 10% of models (number {}):'.format(len(results_bottom10)))
for param in ['kernel', 'gamma']:
    for value in results_bottom10[param].unique():
        print(f'{param} = {value}: {len(results_bottom10[results_bottom10[param] == value])/len(results_bottom10)}')

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/regression/svm/svm_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/regression/svm/svm_hyperparameters.json', 'w') as f:
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

train_preds.to_csv('outputs/regression/svm/svm_train_preds.csv')
test_preds.to_csv('outputs/regression/svm/svm_test_preds.csv')

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
with open('outputs/regression/svm/svm_scores.json', 'w') as f:
    json.dump(scores, f)

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/regression/svm/svm_train_predictions.png')
plt.show()