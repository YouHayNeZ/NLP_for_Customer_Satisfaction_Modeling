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
import lightgbm as lgb
from scipy.stats import randint, uniform
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv', classification=False)

    # Load models
    knn = joblib.load('outputs/predictive_modeling/regression/base_learners/knn/knn_model.pkl')
    rf = joblib.load('outputs/predictive_modeling/regression/base_learners/rf/rf_model.pkl')
    svm = joblib.load('outputs/predictive_modeling/regression/base_learners/svm/svm_model.pkl')
    bayesian_ridge = joblib.load('outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_model.pkl')
    mlp = joblib.load('outputs/predictive_modeling/regression/base_learners/mlp/mlp_model.pkl')



    # # Create ensemble (unweighted)
    # ensemble_unweighted = VotingRegressor(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('bayesian_ridge', bayesian_ridge), ('mlp', mlp)], n_jobs=-1)
    # ensemble_unweighted.fit(X_train, y_train)
    # joblib.dump(ensemble_unweighted, 'outputs/predictive_modeling/regression/ensemble/ensemble_unweighted_model.pkl')

    # # Predictions
    # y_pred_unweighted = np.round(ensemble_unweighted.predict(X_test) + 1)

    # # Metrics
    # mae_unweighted = mean_absolute_error(y_test, y_pred_unweighted)
    # mse_unweighted = mean_squared_error(y_test, y_pred_unweighted)
    # r2_unweighted = r2_score(y_test, y_pred_unweighted)
    # metrics_unweighted = {
    #     'mae': mae_unweighted,
    #     'mse': mse_unweighted,
    #     'r2': r2_unweighted
    # }
    # with open('outputs/predictive_modeling/regression/ensemble/ensemble_unweighted_metrics.json', 'w') as f:
    #     json.dump(metrics_unweighted, f)
    # print(metrics_unweighted)






    # Create ensemble (weighted)
    mae_losses = []

    while len(mae_losses) < 2000:
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
        print("finished iteration ", len(mae_losses))


    # Convert the list to a DataFrame
    mae_losses_df = pd.DataFrame(mae_losses, columns=['w_knn', 'w_rf', 'w_svm', 'w_bayesian_ridge', 'w_mlp', 'mae'])

    # Use parallel coordinate plot for weights and mae scores
    plt.figure(figsize=(14, 7))
    parallel_coordinates(mae_losses_df, 'mae', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/ensemble/ensemble_parallel_coordinates.png')
    plt.close()

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







    # Stacking models (using LightGBM)

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

    # Preprocess stacking data
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv', classification=False)

    # Add knn, rf, svm, nb, mlp predictions as features to X_train from X_train_stacking
    for model in ['knn', 'rf', 'svm', 'bayesian_ridge', 'mlp']:
        X_train[model] = X_train_stacking[model]
        X_val[model] = X_val_stacking[model]
        X_test[model] = X_test_stacking[model]

    # Define the range of hyperparameters
    param_dist = {
        'n_estimators': randint(930, 1250),
        'learning_rate': uniform(0.04, 0.004),  
        'max_depth': randint(20, 23),  
        'num_leaves': randint(7, 9),  
        'reg_lambda': uniform(5.0, 1.3),  
        'min_child_weight': randint(27, 29),
        'feature_fraction': uniform(0.52, 0.025),
        'bagging_fraction': uniform(0.711, 0.001),
        'bagging_freq': randint(3, 4)
    }

    # Hyperparameter tuning
    random_search = RandomizedSearchCV(estimator=lgb.LGBMRegressor(objective='regression', early_stopping_round=5, metric='l1'),
                                param_distributions=param_dist, 
                                n_iter=500, 
                                cv=10, 
                                verbose=2, 
                                scoring='neg_mean_absolute_error',
                                random_state=42, 
                                n_jobs=-1)
    random_search.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)]
                    )

    # Save results of RandomizedSearchCV
    results = pd.DataFrame(random_search.cv_results_)
    interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[interested_columns]
    results = results.sort_values(by='rank_test_score')
    results['mean_test_score'] = results['mean_test_score']
    results.to_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_cv_results.csv')

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_num_leaves': 'num_leaves', 'param_reg_lambda': 'reg_lambda', 'param_min_child_weight': 'min_child_weight', 'param_feature_fraction': 'feature_fraction', 'param_bagging_fraction': 'bagging_fraction', 'param_bagging_freq': 'bagging_freq'})
    results = results.dropna(subset=['mean_test_score'])
    for param in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_lambda', 'min_child_weight', 'feature_fraction', 'bagging_fraction', 'bagging_freq']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results = results.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test,
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_model.pkl',
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_hyperparameters.json',
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_train_preds.csv',
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test, test_preds, 'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_scores.json')

    # Create feature importance plot for top 15
    feature_importance = best_model.feature_importances_
    features = X_train.columns
    feature_importance_scores = dict(zip(features, feature_importance))
    feature_importance_df = pd.DataFrame(feature_importance_scores.items(), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_feature_importance.csv', index=False)

    feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.savefig('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_feature_importance.png')
    plt.close()

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds, 'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_predictions.png')







    # Stacking models (using XGBoost) - base models only without any ryanair data
    # Define the range of hyperparameters
    param_dist2 = {
        'n_estimators': randint(400, 2000),
        'learning_rate': uniform(0.25, 0.15),  
        'max_depth': randint(4, 47),  
        'num_leaves': randint(4, 200),  
        'reg_lambda': uniform(0, 10.0),  
        'min_child_weight': randint(4, 24),
        'feature_fraction': uniform(0.9, 0.1),
        'bagging_fraction': uniform(0.8, 0.2),
        'bagging_freq': randint(0, 10)
    }

    # Hyperparameter tuning
    random_search2 = RandomizedSearchCV(estimator=lgb.LGBMRegressor(objective='regression', early_stopping_round=5, metric='l1'), 
                                param_distributions=param_dist2, 
                                n_iter=1000, 
                                cv=10, 
                                verbose=2, 
                                scoring='neg_mean_absolute_error',
                                random_state=42, 
                                n_jobs=-1)
    random_search2.fit(X_train_stacking, y_train,
                        eval_set=[(X_val_stacking, y_val)]
                    )

    # Save results of RandomizedSearchCV
    results2 = pd.DataFrame(random_search2.cv_results_)
    interested_columns2 = ['param_' + param for param in param_dist2.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results2 = results2[interested_columns2]
    results2 = results2.sort_values(by='rank_test_score')
    results2['mean_test_score'] = results2['mean_test_score']
    results2.to_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_cv_results.csv')

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results2 = results2.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_num_leaves': 'num_leaves', 'param_reg_lambda': 'reg_lambda', 'param_min_child_weight': 'min_child_weight', 'param_feature_fraction': 'feature_fraction', 'param_bagging_fraction': 'bagging_fraction', 'param_bagging_freq': 'bagging_freq'})
    results2 = results2.dropna(subset=['mean_test_score'])
    for param in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_lambda', 'min_child_weight', 'feature_fraction', 'bagging_fraction', 'bagging_freq']:
        results2[param] = scaler.fit_transform(results2[param].values.reshape(-1, 1))
    results2 = results2.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results2, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model2, train_preds2, test_preds2, y_train2, y_test2 = best_model_and_predictions(random_search2, X_train_stacking, X_test_stacking, y_train, y_test, datetime_train, datetime_test,
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_model.pkl',
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_hyperparameters.json',
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_train_preds.csv',
                                'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_test_preds.csv')

    # Regression metrics
    regression_metrics(y_test2, test_preds2, 'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_scores.json')

    # Create feature importance plot for top 15
    feature_importance2 = best_model2.feature_importances_
    features2 = X_train_stacking.columns
    feature_importance_scores2 = dict(zip(features2, feature_importance2))
    feature_importance_df2 = pd.DataFrame(feature_importance_scores2.items(), columns=['Feature', 'Importance'])
    feature_importance_df2 = feature_importance_df2.sort_values(by='Importance', ascending=False)
    feature_importance_df2.to_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_feature_importance.csv', index=False)

    feature_importance_scores2 = dict(sorted(feature_importance_scores2.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores2.keys(), feature_importance_scores2.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.savefig('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_feature_importance.png')
    plt.close()

    # Plot predictions vs real values over time (only regression)
    test_preds_vs_real_over_time(test_preds2, 'outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_predictions.png')

if __name__ == '__main__':
    main()