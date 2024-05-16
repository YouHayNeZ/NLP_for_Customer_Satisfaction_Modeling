import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from preprocessing import *
from sklearn.model_selection import train_test_split

# Load test_preds from all models
xgb_preds = pd.read_csv('outputs/classification/xgboost/xgboost_test_preds.csv')
rf_preds = pd.read_csv('outputs/classification/rf/rf_test_preds.csv')
svm_preds = pd.read_csv('outputs/classification/svm/svm_test_preds.csv')
knn_preds = pd.read_csv('outputs/classification/knn/knn_test_preds.csv')

# Extract and rename columns Predicted Overall Rating to SVM Predictions, XGB Predictions, RF Predictions, KNN Predictions
xgb_preds.rename(columns={'Predicted Overall Rating': 'XGB Predictions'}, inplace=True)
rf_preds.rename(columns={'Predicted Overall Rating': 'RF Predictions'}, inplace=True)
svm_preds.rename(columns={'Predicted Overall Rating': 'SVM Predictions'}, inplace=True)
knn_preds.rename(columns={'Predicted Overall Rating': 'KNN Predictions'}, inplace=True)

rf_predictions = rf_preds['RF Predictions']
svm_predictions = svm_preds['SVM Predictions']
knn_predictions = knn_preds['KNN Predictions']

# Concatenate all predictions column wise
ensemble_preds = pd.concat([xgb_preds, rf_predictions, svm_predictions, knn_predictions], axis=1)

# Convert all predictions to integers (except the date column)
ensemble_preds['Real Overall Rating'] = ensemble_preds['Real Overall Rating'].astype(int)
ensemble_preds['XGB Predictions'] = ensemble_preds['XGB Predictions'].astype(int)
ensemble_preds['RF Predictions'] = ensemble_preds['RF Predictions'].astype(int)
ensemble_preds['SVM Predictions'] = ensemble_preds['SVM Predictions'].astype(int)
ensemble_preds['KNN Predictions'] = ensemble_preds['KNN Predictions'].astype(int)

# Add column 'Ensemble Predictions' with majority voting
ensemble_without_date_and_real = ensemble_preds.drop(columns=['Date Flown', 'Real Overall Rating'])
ensemble_preds['Ensemble Predictions (Unweighted Majority Voting)'] = ensemble_without_date_and_real.mode(axis=1)[0]

# Calculate the accuracy, precision, recall, f1-score and AUC of the ensemble model and compare it to the other models
y_true = ensemble_preds['Real Overall Rating']
y_pred = ensemble_preds['Ensemble Predictions (Unweighted Majority Voting)']
print(np.unique(y_true))
print(np.unique(y_pred))

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1)
roc_auc = auc(fpr, tpr)

print(f'Ensemble Model Accuracy: {accuracy}')
print(f'Ensemble Model Precision: {precision}')
print(f'Ensemble Model Recall: {recall}')
print(f'Ensemble Model F1-Score: {f1}')
print(f'Ensemble Model AUC: {roc_auc}')
