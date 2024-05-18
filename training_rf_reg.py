# Random Forest (Regression)

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from preprocessing import *
from scipy.stats import randint
import joblib
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters
param_dist = {
    'n_estimators': randint(10, 1500),
    'max_features': ["sqrt", "log2", None],
    'max_depth': randint(1, 200),
    'min_samples_split': randint(2, 40),
    'min_samples_leaf': randint(1, 25),
    'bootstrap': [True, False]
}

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=RandomForestRegressor(), 
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
results.to_csv('outputs/regression/rf/rf_cv_results.csv')

# Parallel coordinate plot without max_features and bootstrap
scaler = MinMaxScaler()
results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_max_features': 'max_features', 'param_max_depth': 'max_depth', 'param_min_samples_split': 'min_samples_split', 'param_min_samples_leaf': 'min_samples_leaf', 'param_bootstrap': 'bootstrap'})
for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
max_features_and_bootstrap = results[['max_features', 'bootstrap']]
results = results.drop(columns=['max_features', 'bootstrap', 'std_test_score', 'rank_test_score'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha=0.25)
plt.legend().remove()
plt.savefig('outputs/regression/rf/rf_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Count the percentage of bootstrap = True/False and max_features = sqrt/log2/None
results['bootstrap'] = max_features_and_bootstrap['bootstrap'].astype(str)
results['max_features'] = max_features_and_bootstrap['max_features'].astype(str)
results = results.sort_values(by='mean_test_score', ascending=False)
results_top10 = results.head(int(len(results) * 0.1))
results_bottom10 = results.tail(int(len(results) * 0.1))

# Print length of top 10% and bottom 10% of models
print('Out of top 10% of models (number {}):'.format(len(results_top10)))
print('bootstrap = True: {}'.format(len(results_top10[results_top10['bootstrap'] == 'True'])/len(results_top10)))
print('bootstrap = False: {}'.format(len(results_top10[results_top10['bootstrap'] == 'False'])/len(results_top10)))
print('max_features = sqrt: {}'.format(len(results_top10[results_top10['max_features'] == 'sqrt'])/len(results_top10)))
print('max_features = log2: {}'.format(len(results_top10[results_top10['max_features'] == 'log2'])/len(results_top10)))
print('max_features = None: {}'.format(len(results_top10[results_top10['max_features'] == 'None'])/len(results_top10)))

print('Out of bottom 10% of models (number {})'.format(len(results_bottom10)))
print('bootstrap = True: {}'.format(len(results_bottom10[results_bottom10['bootstrap'] == 'True'])/len(results_bottom10)))
print('bootstrap = False: {}'.format(len(results_bottom10[results_bottom10['bootstrap'] == 'False'])/len(results_bottom10)))
print('max_features = sqrt: {}'.format(len(results_bottom10[results_bottom10['max_features'] == 'sqrt'])/len(results_bottom10)))
print('max_features = log2: {}'.format(len(results_bottom10[results_bottom10['max_features'] == 'log2'])/len(results_bottom10)))
print('max_features = None: {}'.format(len(results_bottom10[results_bottom10['max_features'] == 'None'])/len(results_bottom10)))

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/regression/rf/rf_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/regression/rf/rf_hyperparameters.json', 'w') as f:
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

train_preds.to_csv('outputs/regression/rf/rf_train_preds.csv')
test_preds.to_csv('outputs/regression/rf/rf_test_preds.csv')

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
with open('outputs/regression/rf/rf_scores.json', 'w') as f:
    json.dump(scores, f)

# Create feature importance plot for top 15
feature_importance = best_model.feature_importances_
features = X_train.columns
feature_importance_scores = dict(zip(features, feature_importance))
with open('outputs/regression/rf/rf_feature_importance.json', 'w') as f:
    json.dump(feature_importance_scores, f)

feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
plt.figure(figsize=(14, 7))
plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
plt.ylabel('Feature Importance')
plt.xlabel('Feature')
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.savefig('outputs/regression/rf/rf_feature_importance.png')
plt.show()

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/regression/rf/rf_train_predictions.png')
plt.show()
