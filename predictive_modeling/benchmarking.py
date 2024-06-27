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
    lgbm_preds_only = pd.read_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_base_only_test_preds.csv')
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

    

    # Calculate Accuracy, Recall, Precision, F1 Score, MAE, MSE, R2 for all test predictions and the real test values
    # Classification base learners





if __name__ == '__main__':
    main()
