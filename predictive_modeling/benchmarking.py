# Benchmarking Analysis

# Goal: Compare all classification models with all regression models to determine which 
# type of model is best for this dataset/the task at hand.

# Means: Use saved csv files to compare models along MAE, MSE, R2 as well as 
# Accuracy, Recall, Precision, and F1 Score. (Log Loss not possible)

import pandas as pd
import joblib
import json
import os
import sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from preprocessing import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import BayesianRidge

def main():
    
    # COMPARISON OF CLASSIFICATION AND REGRESSION MODELS (ORIGINAL DATA + ENGINEERED FEATURES)
    # Load csv files
    # Classification base learners
    svm_classif = pd.read_csv('outputs/predictive_modeling/classification/base_learners/svm/svm_test_preds.csv')
    rf_classif = pd.read_csv('outputs/predictive_modeling/classification/base_learners/rf/rf_test_preds.csv')
    mlp_classif = pd.read_csv('outputs/predictive_modeling/classification/base_learners/mlp/mlp_test_preds.csv')
    knn_classif = pd.read_csv('outputs/predictive_modeling/classification/base_learners/knn/knn_test_preds.csv')
    bayes_classif = pd.read_csv('outputs/predictive_modeling/classification/base_learners/nb/nb_test_preds.csv')

    # Classication ensembles
    voting_uw = pd.read_csv('outputs/predictive_modeling/classification/ensemble/ensemble_unweighted_test_preds.csv') 
    voting_w = pd.read_csv('outputs/predictive_modeling/classification/ensemble/ensemble_weighted_test_preds.csv')
    lgbm_preds_only_classif = pd.read_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_test_preds.csv')
    lgbm_all_classif = pd.read_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_test_preds.csv')

    # Regression base learners
    svm_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/svm/svm_test_preds.csv')
    rf_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/rf/rf_test_preds.csv')
    mlp_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/mlp/mlp_test_preds.csv')
    knn_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/knn/knn_test_preds.csv')
    bayes_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_test_preds.csv')

    # Regression ensembles
    avg_uw = pd.read_csv('outputs/predictive_modeling/regression/ensemble/ensemble_unweighted_test_preds.csv')
    avg_w = pd.read_csv('outputs/predictive_modeling/regression/ensemble/ensemble_weighted_test_preds.csv')
    lgbm_preds_only_reg = pd.read_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_test_preds.csv')
    lgbm_all_reg = pd.read_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_test_preds.csv')
    
    # Calculate Accuracy, Recall, Precision, F1 Score, MAE, MSE, R2 for all test predictions and the real test values
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MSE', 'R2', 'MAE'])
    models = ['SVM', 'Random Forest', 'MLP', 'KNN', 'Naive Bayes', 'Voting (Unweighted)', 'Voting (Weighted)', 'LGBM (Only Predictions)', 'LGBM (All Features)', 'SVM', 'Random Forest', 'MLP', 'KNN', 'Bayesian Ridge', 'Average (Unweighted)', 'Average (Weighted)', 'LGBM (Only Predictions)', 'LGBM (All Features)']
    for model in [svm_classif, rf_classif, mlp_classif, knn_classif, bayes_classif, voting_uw, voting_w, lgbm_preds_only_classif, lgbm_all_classif, svm_reg, rf_reg, mlp_reg, knn_reg, bayes_reg, avg_uw, avg_w, lgbm_preds_only_reg, lgbm_all_reg]:
        results = results._append({'Model': models.pop(0),
                                  'Accuracy': accuracy_score(model['Real Overall Rating'], model['Predicted Overall Rating']), 
                                  'F1 Score': f1_score(model['Real Overall Rating'], model['Predicted Overall Rating'], average='weighted'),
                                  'Precision': precision_score(model['Real Overall Rating'], model['Predicted Overall Rating'], average='weighted'), 
                                  'Recall': recall_score(model['Real Overall Rating'], model['Predicted Overall Rating'], average='weighted'), 
                                  'MSE': mean_squared_error(model['Real Overall Rating'], model['Predicted Overall Rating']), 
                                  'R2': r2_score(model['Real Overall Rating'], model['Predicted Overall Rating']), 
                                  'MAE': mean_absolute_error(model['Real Overall Rating'], model['Predicted Overall Rating'])}, ignore_index=True)
    results.to_csv('outputs/predictive_modeling/model_comparison_classif_vs_reg.csv', index=False)

    



    # COMPARISON OF CLASSIFICATION AND REGRESSION MODELS (ORIGINAL DATA ONLY)
    # Load hyperparameters from json files
    # Classification base learners
    with open('outputs/predictive_modeling/classification/base_learners/svm/svm_hyperparameters.json') as f:
        svm_classif_hp = json.load(f)
    with open('outputs/predictive_modeling/classification/base_learners/rf/rf_hyperparameters.json') as f:
        rf_classif_hp = json.load(f)
    with open('outputs/predictive_modeling/classification/base_learners/mlp/mlp_hyperparameters.json') as f:
        mlp_classif_hp = json.load(f)
    with open('outputs/predictive_modeling/classification/base_learners/knn/knn_hyperparameters.json') as f:
        knn_classif_hp = json.load(f)
    with open('outputs/predictive_modeling/classification/base_learners/nb/nb_hyperparameters.json') as f:
        bayes_classif_hp = json.load(f)
    
    # Classication ensembles
    with open('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_hyperparameters.json') as f:
        lgbm_preds_only_classif_hp = json.load(f)
    with open('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_hyperparameters.json') as f:
        lgbm_all_classif_hp = json.load(f)

    # Regression base learners
    with open('outputs/predictive_modeling/regression/base_learners/svm/svm_hyperparameters.json') as f:
        svm_reg_hp = json.load(f)
    with open('outputs/predictive_modeling/regression/base_learners/rf/rf_hyperparameters.json') as f:
        rf_reg_hp = json.load(f)
    with open('outputs/predictive_modeling/regression/base_learners/mlp/mlp_hyperparameters.json') as f:
        mlp_reg_hp = json.load(f)
    with open('outputs/predictive_modeling/regression/base_learners/knn/knn_hyperparameters.json') as f:
        knn_reg_hp = json.load(f)
    with open('outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_hyperparameters.json') as f:
        bayes_reg_hp = json.load(f)
    
    # Regression ensembles
    with open('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_hyperparameters.json') as f:
        lgbm_preds_only_reg_hp = json.load(f)
    with open('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_hyperparameters.json') as f:
        lgbm_all_reg_hp = json.load(f)

    # Load original data
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv', feature_selection=False, classification=False, original_data=True)

    # Retrain all models with best hyperparameters on X_train and y_train
    # Classification base learners
    svm_classif = SVC(**svm_classif_hp).fit(X_train, y_train)
    rf_classif = RandomForestClassifier(**rf_classif_hp).fit(X_train, y_train)
    mlp_classif = MLPClassifier(**mlp_classif_hp).fit(X_train, y_train)
    knn_classif = KNeighborsClassifier(**knn_classif_hp).fit(X_train, y_train)
    bayes_classif = MultinomialNB(**bayes_classif_hp).fit(X_train, y_train)

    # Create data frame from training predictions of all models
    train_preds_classif = pd.DataFrame(columns=['SVM Classification No FE', 'Random Forest Classification No FE', 'MLP Classification No FE', 'KNN Classification No FE', 'Naive Bayes Classification No FE'])
    train_preds_classif['SVM Classification No FE'] = svm_classif.predict(X_train)
    train_preds_classif['Random Forest Classification No FE'] = rf_classif.predict(X_train)
    train_preds_classif['MLP Classification No FE'] = mlp_classif.predict(X_train)
    train_preds_classif['KNN Classification No FE'] = knn_classif.predict(X_train)
    train_preds_classif['Naive Bayes Classification No FE'] = bayes_classif.predict(X_train)

    val_preds_classif = pd.DataFrame(columns=['SVM Classification No FE', 'Random Forest Classification No FE', 'MLP Classification No FE', 'KNN Classification No FE', 'Naive Bayes Classification No FE'])
    val_preds_classif['SVM Classification No FE'] = svm_classif.predict(X_val)
    val_preds_classif['Random Forest Classification No FE'] = rf_classif.predict(X_val)
    val_preds_classif['MLP Classification No FE'] = mlp_classif.predict(X_val)
    val_preds_classif['KNN Classification No FE'] = knn_classif.predict(X_val)
    val_preds_classif['Naive Bayes Classification No FE'] = bayes_classif.predict(X_val)

    # Regression base learners
    svm_reg = SVR(**svm_reg_hp).fit(X_train, y_train)
    rf_reg = RandomForestRegressor(**rf_reg_hp).fit(X_train, y_train)
    mlp_reg = MLPRegressor(**mlp_reg_hp).fit(X_train, y_train)
    knn_reg = KNeighborsRegressor(**knn_reg_hp).fit(X_train, y_train)
    bayes_reg = BayesianRidge(**bayes_reg_hp).fit(X_train, y_train)

    # Create data frame from training predictions of all models
    train_preds_reg = pd.DataFrame(columns=['SVM Regression No FE', 'Random Forest Regression No FE', 'MLP Regression No FE', 'KNN Regression No FE', 'Bayesian Ridge Regression No FE'])
    train_preds_reg['SVM Regression No FE'] = svm_reg.predict(X_train)
    train_preds_reg['Random Forest Regression No FE'] = rf_reg.predict(X_train)
    train_preds_reg['MLP Regression No FE'] = mlp_reg.predict(X_train)
    train_preds_reg['KNN Regression No FE'] = knn_reg.predict(X_train)
    train_preds_reg['Bayesian Ridge Regression No FE'] = bayes_reg.predict(X_train)

    val_preds_reg = pd.DataFrame(columns=['SVM Regression No FE', 'Random Forest Regression No FE', 'MLP Regression No FE', 'KNN Regression No FE', 'Bayesian Ridge Regression No FE'])
    val_preds_reg['SVM Regression No FE'] = svm_reg.predict(X_val)
    val_preds_reg['Random Forest Regression No FE'] = rf_reg.predict(X_val)
    val_preds_reg['MLP Regression No FE'] = mlp_reg.predict(X_val)
    val_preds_reg['KNN Regression No FE'] = knn_reg.predict(X_val)
    val_preds_reg['Bayesian Ridge Regression No FE'] = bayes_reg.predict(X_val)

    # Reset index of X_train, X_val, train_preds_classif, val_preds_classif, train_preds_reg, val_preds_reg
    X_train = X_train.reset_index(drop=True)
    X_val = X_val.reset_index(drop=True)
    train_preds_classif = train_preds_classif.reset_index(drop=True)
    val_preds_classif = val_preds_classif.reset_index(drop=True)
    train_preds_reg = train_preds_reg.reset_index(drop=True)
    val_preds_reg = val_preds_reg.reset_index(drop=True)

    # Combine X_train with train_preds_classif (and same for validation)
    X_train_ensemble_classif = pd.concat([X_train, train_preds_classif], axis=1)
    X_val_ensemble_classif = pd.concat([X_val, val_preds_classif], axis=1)

    # Combine X_train with train_preds_reg (and same for validation)
    X_train_ensemble_reg = pd.concat([X_train, train_preds_reg], axis=1)
    X_val_ensemble_reg = pd.concat([X_val, val_preds_reg], axis=1)

    # Classication ensembles
    lgbm_preds_only_classif = LGBMClassifier(metric='multi_logloss', early_stopping_rounds=1, objective='multiclass', **lgbm_preds_only_classif_hp).fit(train_preds_classif, y_train, eval_set=[(val_preds_classif, y_val)])
    lgbm_all_classif = LGBMClassifier(metric='multi_logloss', early_stopping_rounds=5, objective='multiclass', **lgbm_all_classif_hp).fit(X_train_ensemble_classif, y_train, eval_set=[(X_val_ensemble_classif, y_val)])

    # Regression ensembles
    lgbm_preds_only_reg = LGBMRegressor(objective='regression', early_stopping_round=1, metric='l1', **lgbm_preds_only_reg_hp).fit(train_preds_reg, y_train, eval_set=[(val_preds_reg, y_val)])
    lgbm_all_reg = LGBMRegressor(objective='regression', early_stopping_round=5, metric='l1', **lgbm_all_reg_hp).fit(X_train_ensemble_reg, y_train, eval_set=[(X_val_ensemble_reg, y_val)])
    
    # Create data frame from test predictions of all models
    base_learner_names = ['SVM Classification No FE', 'Random Forest Classification No FE', 'MLP Classification No FE', 'KNN Classification No FE', 'Naive Bayes Classification No FE', 'SVM Regression No FE', 'Random Forest Regression No FE', 'MLP Regression No FE', 'KNN Regression No FE', 'Bayesian Ridge Regression No FE']
    test_preds = pd.DataFrame()
    base_learners = [svm_classif, rf_classif, mlp_classif, knn_classif, bayes_classif, svm_reg, rf_reg, mlp_reg, knn_reg, bayes_reg]

    for model_name, model in zip(base_learner_names, base_learners):
        test_preds[model_name] = np.round(model.predict(X_test))
    
    X_test = X_test.reset_index(drop=True)
    test_preds = test_preds.reset_index(drop=True)
    X_test_ensemble_classif = pd.concat([X_test, test_preds.iloc[:, 0:5]], axis=1)
    X_test_ensemble_reg = pd.concat([X_test, test_preds.iloc[:, 5:10]], axis=1)
    
    test_preds['LGBM (Only Predictions) Classification No FE'] = np.round(lgbm_preds_only_classif.predict(test_preds.iloc[:, 0:5]))
    test_preds['LGBM (Only Predictions) Regression No FE'] = np.round(lgbm_preds_only_reg.predict(test_preds.iloc[:, 5:10]))
    test_preds['LGBM (All Features) Classification No FE'] = np.round(lgbm_all_classif.predict(X_test_ensemble_classif))
    test_preds['LGBM (All Features) Regression No FE'] = np.round(lgbm_all_reg.predict(X_test_ensemble_reg))

    y_test = y_test.reset_index(drop=True)
    test_preds = test_preds.reset_index(drop=True)
    test_preds['Real Overall Rating'] = y_test

    # Calculate Accuracy, Recall, Precision, F1 Score, MAE, MSE, R2 for all test predictions and the real test values
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall', 'MSE', 'R2', 'MAE'])
    models = ['SVM Classification No FE', 'Random Forest Classification No FE', 'MLP Classification No FE', 'KNN Classification No FE', 'Naive Bayes Classification No FE', 'LGBM (Only Predictions) Classification No FE', 'LGBM (All Features) Classification No FE', 'SVM Regression No FE', 'Random Forest Regression No FE', 'MLP Regression No FE', 'KNN Regression No FE', 'Bayesian Ridge Regression No FE', 'LGBM (Only Predictions) Regression No FE', 'LGBM (All Features) Regression No FE']
    for model in [svm_classif, rf_classif, mlp_classif, knn_classif, bayes_classif, lgbm_preds_only_classif, lgbm_all_classif, svm_reg, rf_reg, mlp_reg, knn_reg, bayes_reg, lgbm_preds_only_reg, lgbm_all_reg]:
        model_name = models.pop(0)
        results = results._append({'Model': model_name,
                                    'Accuracy': accuracy_score(test_preds[model_name], test_preds['Real Overall Rating']),
                                    'F1 Score': f1_score(test_preds[model_name], test_preds['Real Overall Rating'], average='weighted'),
                                    'Precision': precision_score(test_preds[model_name], test_preds['Real Overall Rating'], average='weighted'),
                                    'Recall': recall_score(test_preds[model_name], test_preds['Real Overall Rating'], average='weighted'),
                                    'MSE': mean_squared_error(test_preds[model_name], test_preds['Real Overall Rating']),
                                    'R2': r2_score(test_preds[model_name], test_preds['Real Overall Rating']),
                                    'MAE': mean_absolute_error(test_preds[model_name], test_preds['Real Overall Rating'])}, ignore_index=True)

    # Save test scores to csv
    results.to_csv('outputs/predictive_modeling/model_comparison_test_preds_with_vs_without_FE.csv', index=False)

if __name__ == '__main__':
    main()
