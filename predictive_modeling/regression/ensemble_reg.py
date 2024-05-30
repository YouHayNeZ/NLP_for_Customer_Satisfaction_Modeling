import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import VotingRegressor
import json
import random
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint, uniform
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

    # Load models
    knn = joblib.load('outputs/predictive_modeling/regression/knn/knn_model.pkl')
    rf = joblib.load('outputs/predictive_modeling/regression/rf/rf_model.pkl')
    svm = joblib.load('outputs/predictive_modeling/regression/svm/svm_model.pkl')
    bayesian_ridge = joblib.load('outputs/predictive_modeling/regression/bayesian_ridge/bayesian_ridge_model.pkl')
    mlp = joblib.load('outputs/predictive_modeling/regression/mlp/mlp_model.pkl')



    # Create ensemble (unweighted)
    ensemble_unweighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge), ('mlp', mlp)], n_jobs=-1)
    ensemble_unweighted.fit(X_train, y_train)
    joblib.dump(ensemble_unweighted, 'outputs/predictive_modeling/regression/ensemble/ensemble_unweighted_model.pkl')

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
    with open('outputs/predictive_modeling/regression/ensemble/ensemble_unweighted_metrics.json', 'w') as f:
        json.dump(metrics_unweighted, f)
    print(metrics_unweighted)






    # Create ensemble (weighted)
    mae_losses = []

    while len(mae_losses) < 10:
        w_rf = random.uniform(0, 1)
        w_knn = random.uniform(0, 1)
        w_svm = random.uniform(0, 1)
        w_bayesian_ridge = random.uniform(0, 1)
        w_mlp = random.uniform(0, 1)
        ensemble_weighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge), ('mlp', mlp)], n_jobs=-1, weights=[w_knn, w_rf, w_svm, w_bayesian_ridge, w_mlp])
        ensemble_weighted.fit(X_train, y_train)
        y_pred = np.round(ensemble_weighted.predict(X_val) + 1)
        mae = mean_absolute_error(y_val, y_pred)
        mae_losses.append([w_knn, w_rf, w_svm, w_bayesian_ridge, w_mlp, mae])

    # Convert the list to a DataFrame
    mae_losses_df = pd.DataFrame(mae_losses, columns=['w_knn', 'w_rf', 'w_svm', 'w_bayesian_ridge', 'w_mlp', 'mae'])

    # Use parallel coordinate plot for weights and mae scores
    plt.figure(figsize=(14, 7))
    parallel_coordinates(mae_losses_df, 'mae', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/ensemble/ensemble_parallel_coordinates.png')
    plt.show()

    mae_scores = mae_losses_df.sort_values(by='mae', ascending=True)
    mae_scores.to_csv('outputs/predictive_modeling/regression/ensemble/ensemble_weighted_mae_scores.csv')
    best_weights = mae_scores.iloc[0, :-1].values

    ensemble_weighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge), ('mlp', mlp)], n_jobs=-1, weights=best_weights)
    ensemble_weighted.fit(X_train, y_train)
    y_pred_weighted = np.round(ensemble_weighted.predict(X_test) + 1)

    joblib.dump(ensemble_weighted, 'outputs/predictive_modeling/regression/ensemble/ensemble_weighted_model.pkl')

    # Metrics
    mae_weighted = mean_absolute_error(y_test, y_pred_weighted)
    mse_weighted = mean_squared_error(y_test, y_pred_weighted)
    r2_weighted = r2_score(y_test, y_pred_weighted)
    metrics_weighted = {
        'mae': mae_weighted,
        'mse': mse_weighted,
        'r2': r2_weighted
    }
    with open('outputs/predictive_modeling/regression/ensemble/ensemble_weighted_metrics.json', 'w') as f:
        json.dump(metrics_weighted, f)
    print(metrics_weighted)







    # Stacking models (using XGBoost)

    # Create data frame with training predictions of all models and real labels
    X_train_stacking = pd.DataFrame({
        'knn': np.round(knn.predict(X_train) + 1),
        'rf': np.round(rf.predict(X_train) + 1),
        'svm': np.round(svm.predict(X_train) + 1),
        'bayesian_ridge': np.round(bayesian_ridge.predict(X_train) + 1),
        'mlp': np.round(mlp.predict(X_train) + 1)
    })

    X_val_stacking = pd.DataFrame({
        'knn': np.round(knn.predict(X_val) + 1),
        'rf': np.round(rf.predict(X_val) + 1),
        'svm': np.round(svm.predict(X_val) + 1),
        'bayesian_ridge': np.round(bayesian_ridge.predict(X_val) + 1),
        'mlp': np.round(mlp.predict(X_val) + 1)
    })

    X_test_stacking = pd.DataFrame({
        'knn': np.round(knn.predict(X_test) + 1),
        'rf': np.round(rf.predict(X_test) + 1),
        'svm': np.round(svm.predict(X_test) + 1),
        'bayesian_ridge': np.round(bayesian_ridge.predict(X_test) + 1),
        'mlp': np.round(mlp.predict(X_test) + 1)
    })

    # Concatenate all 3 data frames
    X_stacked = pd.concat([X_train_stacking, X_val_stacking, X_test_stacking])
    X_stacked = X_stacked.reset_index(drop=True)

    # Load original data
    data = pd.read_csv('data/ryanair_reviews.csv')
    data = data.dropna(subset=['Overall Rating'])
    data = pd.concat([data, X_stacked], axis=1)
    data.to_csv('outputs/predictive_modeling/regression/ensemble/stacking_data.csv', index=False)

    # Preprocess stacking data
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('outputs/predictive_modeling/regression/ensemble/stacking_data.csv')

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
                                n_iter=100, 
                                cv=5, 
                                verbose=2, 
                                scoring='neg_mean_absolute_error',
                                random_state=42, 
                                n_jobs=-1)
    random_search.fit(X_train, y_train,
                        early_stopping_rounds=20,
                        eval_set=[(X_val, y_val)],
                        eval_metric='mae',
                    )

    # Save results of RandomizedSearchCV
    results = pd.DataFrame(random_search.cv_results_)
    interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[interested_columns]
    results = results.sort_values(by='rank_test_score')
    results['mean_test_score'] = results['mean_test_score']
    results.to_csv('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_cv_results.csv')

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_min_child_weight': 'min_child_weight', 'param_colsample_bytree': 'colsample_bytree', 'param_colsample_bylevel': 'colsample_bylevel', 'param_reg_lambda': 'reg_lambda', 'param_reg_alpha': 'reg_alpha', 'param_scale_pos_weight': 'scale_pos_weight', 'param_gamma': 'gamma'})
    for param in ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'colsample_bytree', 'colsample_bylevel', 'reg_lambda', 'reg_alpha', 'scale_pos_weight', 'gamma']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results = results.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_parallel_coordinates.png')
    plt.show()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test,
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_model.pkl',
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_hyperparameters.json',
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_train_preds.csv',
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_scores.json')

    # Create feature importance plot for top 15
    feature_importance = best_model.feature_importances_
    features = X_train.columns
    feature_importance_scores = dict(zip(features, feature_importance))
    feature_importance_df = pd.DataFrame(feature_importance_scores.items(), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_feature_importance.csv', index=False)

    feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.savefig('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_feature_importance.png')
    plt.show()

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_predictions.png')







    # Stacking models (using XGBoost) - base models only without any ryanair data
    # Define the range of hyperparameters
    param_dist2 = {
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
    random_search2 = RandomizedSearchCV(estimator=xgb.XGBRegressor(), 
                                param_distributions=param_dist2, 
                                n_iter=100, 
                                cv=5, 
                                verbose=2, 
                                scoring='neg_mean_absolute_error',
                                random_state=42, 
                                n_jobs=-1)
    random_search2.fit(X_train_stacking, y_train,
                        early_stopping_rounds=20,
                        eval_set=[(X_val_stacking, y_val)],
                        eval_metric='mae',
                    )

    # Save results of RandomizedSearchCV
    results2 = pd.DataFrame(random_search2.cv_results_)
    interested_columns2 = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results2 = results2[interested_columns2]
    results2 = results2.sort_values(by='rank_test_score')
    results['mean_test_score'] = results2['mean_test_score']
    results2.to_csv('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_cv_results.csv')

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results2 = results2.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_min_child_weight': 'min_child_weight', 'param_colsample_bytree': 'colsample_bytree', 'param_colsample_bylevel': 'colsample_bylevel', 'param_reg_lambda': 'reg_lambda', 'param_reg_alpha': 'reg_alpha', 'param_scale_pos_weight': 'scale_pos_weight', 'param_gamma': 'gamma'})
    for param in ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'colsample_bytree', 'colsample_bylevel', 'reg_lambda', 'reg_alpha', 'scale_pos_weight', 'gamma']:
        results2[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results2 = results2.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results2, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_parallel_coordinates.png')
    plt.show()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model2, train_preds2, test_preds2, y_train2, y_test2 = best_model_and_predictions(random_search2, X_train_stacking, X_test_stacking, y_train, y_test, datetime_train, datetime_test,
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_model.pkl',
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_hyperparameters.json',
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_train_preds.csv',
                                'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test2, test_preds2, 'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_scores.json')

    # Create feature importance plot for top 15
    feature_importance2 = best_model2.feature_importances_
    features2 = X_train_stacking.columns
    feature_importance_scores2 = dict(zip(features2, feature_importance2))
    feature_importance_df2 = pd.DataFrame(feature_importance_scores2.items(), columns=['Feature', 'Importance'])
    feature_importance_df2 = feature_importance_df2.sort_values(by='Importance', ascending=False)
    feature_importance_df2.to_csv('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_feature_importance.csv', index=False)

    feature_importance_scores2 = dict(sorted(feature_importance_scores2.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores2.keys(), feature_importance_scores2.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.savefig('outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_feature_importance.png')
    plt.show()

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds2, 'outputs/predictive_modeling/regression/ensemble/xgboost_ensemble_base_only_predictions.png')

if __name__ == '__main__':
    main()