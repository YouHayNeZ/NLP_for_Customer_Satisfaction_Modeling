import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
from preprocessing import create_pipeline
from sklearn.model_selection import train_test_split
import json
import random
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint, uniform

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Load models
knn = joblib.load('outputs/regression/knn/knn_model.pkl')
rf = joblib.load('outputs/regression/rf/rf_model.pkl')
svm = joblib.load('outputs/regression/svm/svm_model.pkl')
bayesian_ridge = joblib.load('outputs/regression/bayesian_ridge/bayesian_ridge_model.pkl')




# Create ensemble (unweighted)
ensemble_unweighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge)], n_jobs=-1)
ensemble_unweighted.fit(X_train, y_train)
joblib.dump(ensemble_unweighted, 'outputs/regression/ensemble/ensemble_unweighted_model.pkl')

# Predictions
y_pred_unweighted = np.round(ensemble_unweighted.predict(X_test) + 1)

# Metrics
mae_unweighted = mean_absolute_error(y_test, y_pred_unweighted)
mse_unweighted = mean_squared_error(y_test, y_pred_unweighted)
r2_unweighted = r2_score(y_test, y_pred_unweighted)
metrics_unweighted = {
    'mae': mae_unweighted,
    'mse': mse_unweighted,
    'r2': r2_unweighted
}
with open('outputs/regression/ensemble/ensemble_unweighted_metrics.json', 'w') as f:
    json.dump(metrics_unweighted, f)
print(metrics_unweighted)






# Create ensemble (weighted)
mae_losses = []

while len(mae_losses) < 30:
    w_rf = random.uniform(0, 1)
    w_knn = random.uniform(0, 1)
    w_svm = random.uniform(0, 1)
    w_bayesian_ridge = random.uniform(0, 1)
    ensemble_weighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge)], n_jobs=-1, weights=[w_knn, w_rf, w_svm, w_bayesian_ridge])
    ensemble_weighted.fit(X_train, y_train)
    y_pred = np.round(ensemble_weighted.predict(X_test) + 1)
    mae = mean_absolute_error(y_test, y_pred)
    mae_losses.append([w_knn, w_rf, w_svm, w_bayesian_ridge, mae])

# Convert the list to a DataFrame
mae_losses_df = pd.DataFrame(mae_losses, columns=['w_knn', 'w_rf', 'w_svm', 'w_bayesian_ridge', 'mae'])

# Use parallel coordinate plot for weights and mae scores
plt.figure(figsize=(14, 7))
parallel_coordinates(mae_losses_df, 'mae', colormap='viridis', alpha=0.25)
plt.legend().remove()
plt.savefig('outputs/regression/ensemble/ensemble_parallel_coordinates.png')
plt.show()

mae_scores = mae_losses_df.sort_values(by='mae', ascending=True)
mae_scores.to_csv('outputs/regression/ensemble/ensemble_weighted_mae_scores.csv')
best_weights = mae_scores.iloc[0, :-1].values

ensemble_weighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge)], n_jobs=-1, weights=best_weights)
ensemble_weighted.fit(X_train, y_train)
y_pred_weighted = np.round(ensemble_weighted.predict(X_test) + 1)

joblib.dump(ensemble_weighted, 'outputs/regression/ensemble/ensemble_weighted_model.pkl')

# Metrics
mae_weighted = mean_absolute_error(y_test, y_pred_weighted)
mse_weighted = mean_squared_error(y_test, y_pred_weighted)
r2_weighted = r2_score(y_test, y_pred_weighted)
metrics_weighted = {
    'mae': mae_weighted,
    'mse': mse_weighted,
    'r2': r2_weighted
}
with open('outputs/regression/ensemble/ensemble_weighted_metrics.json', 'w') as f:
    json.dump(metrics_weighted, f)
print(metrics_weighted)







# Stacking models (using XGBoost)

# Create data frame with training predictions of all models and real labels
X_train_stacking = pd.DataFrame({
    'knn': np.round(knn.predict(X_train) + 1),
    'rf': np.round(rf.predict(X_train) + 1),
    'svm': np.round(svm.predict(X_train) + 1),
    'bayesian_ridge': np.round(bayesian_ridge.predict(X_train) + 1)
})

X_val_stacking = pd.DataFrame({
    'knn': np.round(knn.predict(X_val) + 1),
    'rf': np.round(rf.predict(X_val) + 1),
    'svm': np.round(svm.predict(X_val) + 1),
    'bayesian_ridge': np.round(bayesian_ridge.predict(X_val) + 1)
})

X_test_stacking = pd.DataFrame({
    'knn': np.round(knn.predict(X_test) + 1),
    'rf': np.round(rf.predict(X_test) + 1),
    'svm': np.round(svm.predict(X_test) + 1),
    'bayesian_ridge': np.round(bayesian_ridge.predict(X_test) + 1)
})

# Concatenate all 3 data frames
X_stacked = pd.concat([X_train_stacking, X_val_stacking, X_test_stacking])
X_stacked = X_stacked.reset_index(drop=True)

# Load original data
data = pd.read_csv('data/ryanair_reviews.csv')
data = data.dropna(subset=['Overall Rating'])
data = pd.concat([data, X_stacked], axis=1)
data.to_csv('outputs/regression/ensemble/stacking_data.csv', index=False)

# Preprocess stacking data
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('outputs/regression/ensemble/stacking_data.csv')

# Define the range of hyperparameters
param_dist = {
    'n_estimators': randint(100, 2000),
    'learning_rate': uniform(0.0001, 0.1),  
    'max_depth': randint(1, 20),  
    'min_child_weight': randint(1, 21),
    'colsample_bytree': uniform(0.0, 1.0),  
    'colsample_bylevel': uniform(0.0, 1.0),
    'reg_lambda': uniform(0.0001, 10.0),  
    'reg_alpha': uniform(0.0001, 1.0), 
    'scale_pos_weight': uniform(0.0, 10.0),  
    'gamma': uniform(0.0001, 10.0)  
}

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=xgb.XGBRegressor(), 
                               param_distributions=param_dist, 
                               n_iter=50, 
                               cv=2, 
                               verbose=2, 
                               scoring='neg_mean_absolute_error',
                               random_state=42, 
                               n_jobs=-1)
random_search.fit(X_train, y_train,
                    early_stopping_rounds=10,
                    eval_set=[(X_val, y_val)],
                    eval_metric='mae',
                  )

# Save results of RandomizedSearchCV
results = pd.DataFrame(random_search.cv_results_)
interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
results = results[interested_columns]
results = results.sort_values(by='rank_test_score')
results['mean_test_score'] = results['mean_test_score']
results.to_csv('outputs/regression/ensemble/xgboost_ensemble_cv_results.csv')

# Parallel coordinate plot without max_features and bootstrap
scaler = MinMaxScaler()
results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_min_child_weight': 'min_child_weight', 'param_colsample_bytree': 'colsample_bytree', 'param_colsample_bylevel': 'colsample_bylevel', 'param_reg_lambda': 'reg_lambda', 'param_reg_alpha': 'reg_alpha', 'param_scale_pos_weight': 'scale_pos_weight', 'param_gamma': 'gamma'})
for param in ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'colsample_bytree', 'colsample_bylevel', 'reg_lambda', 'reg_alpha', 'scale_pos_weight', 'gamma']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results = results.drop(columns=['std_test_score', 'rank_test_score'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha = 0.25)
plt.legend().remove()
plt.savefig('outputs/regression/ensemble/xgboost_ensemble_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/regression/ensemble/xgboost_ensemble_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/regression/ensemble/xgboost_ensemble_hyperparameters.json', 'w') as f:
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

train_preds.to_csv('outputs/regression/ensemble/xgboost_train_preds.csv')
test_preds.to_csv('outputs/regression/ensemble/xgboost_test_preds.csv')

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
with open('outputs/regression/ensemble/xgboost_scores.json', 'w') as f:
    json.dump(scores, f)

# Create feature importance plot for top 15
feature_importance = best_model.feature_importances_
features = X_train.columns
feature_importance_scores = dict(zip(features, feature_importance))
feature_importance_df = pd.DataFrame(feature_importance_scores.items(), columns=['Feature', 'Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.to_csv('outputs/regression/ensemble/xgboost_ensemble_feature_importance.csv', index=False)

feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
plt.figure(figsize=(14, 7))
plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
plt.ylabel('Feature Importance')
plt.xlabel('Feature')
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.savefig('outputs/regression/ensemble/xgboost_ensemble_feature_importance.png')
plt.show()

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/regression/ensemble/xgboost_ensemble_train_predictions.png')
plt.show()