# Benchmarking Analysis

# Goal: Compare all classification models with all regression models to determine which 
# type of model is best for this dataset/the task at hand.

# Means: Use saved csv files to compare models along MAE, MSE, R2 as well as 
# Accuracy, Recall, Precision, and F1 Score. (Log Loss not possible)

import pandas as pd
import joblib
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from preprocessing import *
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

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
    lgbm_preds_only = pd.read_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_test_preds.csv')
    lgbm_all = pd.read_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_test_preds.csv')

    # Regression base learners
    svm_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/svm/svm_test_preds.csv')
    rf_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/rf/rf_test_preds.csv')
    mlp_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/mlp/mlp_test_preds.csv')
    knn_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/knn/knn_test_preds.csv')
    bayes_reg = pd.read_csv('outputs/predictive_modeling/regression/base_learners/bayesian_ridge/bayesian_ridge_test_preds.csv')

    # Regression ensembles
    avg_uw = pd.read_csv('outputs/predictive_modeling/regression/ensemble/ensemble_unweighted_test_preds.csv')
    avg_w = pd.read_csv('outputs/predictive_modeling/regression/ensemble/ensemble_weighted_test_preds.csv')
    lgbm_preds_only = pd.read_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_base_only_test_preds.csv')
    lgbm_all = pd.read_csv('outputs/predictive_modeling/regression/ensemble/lgbm_ensemble_test_preds.csv')

    # Add columns Overall Rating and Date Published to voting_uw, voting_w, avg_uw, avg_w
    for model in [voting_uw, voting_w, avg_uw, avg_w]:
        model['Date Published'] = datetime_test
        model['Overall Rating'] = y_test
    
    # Calculate Accuracy, Recall, Precision, F1 Score, MAE, MSE, R2 for all test predictions and the real test values
    results = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'Precision', 'F1 Score', 'MAE', 'MSE', 'R2'])
    models = ['SVM', 'Random Forest', 'MLP', 'KNN', 'Naive Bayes', 'Voting (Unweighted)', 'Voting (Weighted)', 'LGBM (Only Predictions)', 'LGBM (All Features)', 'SVM', 'Random Forest', 'MLP', 'KNN', 'Naive Bayes', 'Average (Unweighted)', 'Average (Weighted)', 'LGBM (Only Predictions)', 'LGBM (All Features)']
    for model in [svm_classif, rf_classif, mlp_classif, knn_classif, bayes_classif, voting_uw, voting_w, lgbm_preds_only, lgbm_all, svm_reg, rf_reg, mlp_reg, knn_reg, bayes_reg, avg_uw, avg_w, lgbm_preds_only, lgbm_all]:
        results = results._append({'Model': models.pop(0),
                                  'Accuracy': accuracy_score(y_test, model['Predicted Overall Rating']), 
                                  'Recall': recall_score(y_test, model['Predicted Overall Rating'], average='weighted'), 
                                  'Precision': precision_score(y_test, model['Predicted Overall Rating'], average='weighted'), 
                                  'F1 Score': f1_score(y_test, model['Predicted Overall Rating'], average='weighted'), 
                                  'MAE': mean_absolute_error(y_test, model['Predicted Overall Rating']), 
                                  'MSE': mean_squared_error(y_test, model['Predicted Overall Rating']), 
                                  'R2': r2_score(y_test, model['Predicted Overall Rating'])}, ignore_index=True)
    results.to_csv('outputs/predictive_modeling/benchmarking_results.csv', index=False)

if __name__ == '__main__':
    main()
