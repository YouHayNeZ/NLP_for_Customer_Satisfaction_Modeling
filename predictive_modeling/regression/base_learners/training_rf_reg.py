# Random Forest (Regression)

# Importing the libraries
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing_revised import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

#     # Define the range of hyperparameters
#     param_dist = {
#         'n_estimators': randint(10, 1500),
#         'max_features': ["sqrt", "log2", None],
#         'max_depth': randint(1, 200),
#         'min_samples_split': randint(2, 40),
#         'min_samples_leaf': randint(1, 25),
#         'bootstrap': [True, False]
#     }

#     # Hyperparameter tuning & CV results
#     random_search, results = hpo_and_cv_results(RandomForestRegressor(), 'outputs/predictive_modeling/regression/base_learners/rf/rf_cv_results.csv', param_dist, X_train, y_train, scoring='neg_mean_absolute_error')

#     # Parallel coordinate plot without max_features and bootstrap
#     scaler = MinMaxScaler()
#     results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_max_features': 'max_features', 'param_max_depth': 'max_depth', 'param_min_samples_split': 'min_samples_split', 'param_min_samples_leaf': 'min_samples_leaf', 'param_bootstrap': 'bootstrap'})
#     for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
#         results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
#     results_pc = results.drop(columns=['max_features', 'bootstrap', 'std_test_score', 'rank_test_score'])
#     plt.figure(figsize=(14, 7))
#     parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
#     plt.legend().remove()
#     plt.savefig('outputs/predictive_modeling/regression/base_learners/rf/rf_parallel_coordinates.png')
#     plt.show()
#     # purple = best, yellow = worst

#     # Percentage of each hyperparameter combination (top 10% and bottom 10%)
#     perc_of_hp_combinations(results, ['max_features', 'bootstrap'])

#     # Best model, hyperparameters and predictions
#     best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
#                             'outputs/predictive_modeling/regression/base_learners/rf/rf_model.pkl', 
#                             'outputs/predictive_modeling/regression/base_learners/rf/rf_hyperparameters.json', 
#                             'outputs/predictive_modeling/regression/base_learners/rf/rf_train_preds.csv', 
#                             'outputs/predictive_modeling/regression/base_learners/rf/rf_test_preds.csv')

#     # Regression metrics
#     regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/base_learners/rf/rf_scores.json')

#     # Plot predictions vs real values over time (only regression)
#     test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/base_learners/rf/rf_train_predictions.png')

#     # Create feature importance plot for top 15
#     feature_importance = best_model.feature_importances_
#     features = X_train.columns
#     feature_importance_scores = dict(zip(features, feature_importance))
#     with open('outputs/predictive_modeling/regression/base_learners/rf/rf_feature_importance.json', 'w') as f:
#         json.dump(feature_importance_scores, f)

#     feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
#     plt.figure(figsize=(14, 7))
#     plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
#     plt.ylabel('Feature Importance')
#     plt.xlabel('Feature')
#     plt.xticks(rotation=90)
#     plt.title('Feature Importance')
#     plt.savefig('outputs/predictive_modeling/regression/base_learners/rf/rf_feature_importance.png')
#     plt.show()

if __name__ == '__main__':
    main()