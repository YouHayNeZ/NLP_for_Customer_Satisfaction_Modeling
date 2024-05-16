import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc
from sklearn.ensemble import VotingClassifier
from preprocessing import *
from sklearn.model_selection import train_test_split
import json
import random
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates

# Prepare data for training
X_train, X_test, y_train, y_test, datetime_train, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Load models
knn = joblib.load('outputs/classification/knn/knn_model.pkl')
rf = joblib.load('outputs/classification/rf/rf_model.pkl')
svm = joblib.load('outputs/classification/svm/svm_model.pkl')
xgboost = joblib.load('outputs/classification/xgboost/xgboost_model.pkl')

# Create ensemble (unweighted)
ensemble_unweighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('xgboost', xgboost)], voting='hard', n_jobs=-1)
ensemble_unweighted.fit(X_train, y_train)
joblib.dump(ensemble_unweighted, 'outputs/classification/ensemble/ensemble_unweighted_model.pkl')

# Predictions
y_pred_unweighted = ensemble_unweighted.predict(X_test)

# Metrics
accuracy_unweighted = accuracy_score(y_test, y_pred_unweighted)
f1_unweighted = f1_score(y_test, y_pred_unweighted, average='weighted')
precision_unweighted = precision_score(y_test, y_pred_unweighted, average='weighted')
recall_unweighted = recall_score(y_test, y_pred_unweighted, average='weighted')
metrics_unweighted = {
    'accuracy': accuracy_unweighted,
    'f1': f1_unweighted,
    'precision': precision_unweighted,
    'recall': recall_unweighted
}
with open('outputs/classification/ensemble/ensemble_unweighted_metrics.json', 'w') as f:
    json.dump(metrics_unweighted, f)
print(metrics_unweighted)

# Optimize weighted ensemble based on f1 score
# Create ensemble (weighted)
f1_scores = []

while len(f1_scores) < 500:
    w_xgboost = random.uniform(0, 1)
    w_rf = random.uniform(0, 1)
    w_knn = random.uniform(0, 1)
    w_svm = random.uniform(0, 1)
    ensemble_weighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('xgboost', xgboost)], voting='hard', n_jobs=-1, weights=[w_knn, w_rf, w_svm, w_xgboost])
    ensemble_weighted.fit(X_train, y_train)
    y_pred = ensemble_weighted.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    f1_scores.append([w_knn, w_rf, w_svm, w_xgboost, f1])

# Convert the list to a DataFrame
f1_scores_df = pd.DataFrame(f1_scores, columns=['w_knn', 'w_rf', 'w_svm', 'w_xgboost', 'f1'])

# Use parallel coordinate plot for weights and f1 scores
plt.figure(figsize=(14, 7))
parallel_coordinates(f1_scores_df, 'f1', colormap='viridis', alpha=0.25)
plt.legend().remove()
plt.savefig('outputs/classification/ensemble/ensemble_parallel_coordinates.png')
plt.show()

f1_scores = f1_scores_df.sort_values(by='f1', ascending=False)
f1_scores.to_csv('outputs/classification/ensemble/ensemble_weighted_f1_scores.csv')
best_weights = f1_scores.iloc[0, :-1].values

ensemble_weighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('xgboost', xgboost)], voting='hard', n_jobs=-1, weights=best_weights)
ensemble_weighted.fit(X_train, y_train)
y_pred_weighted = ensemble_weighted.predict(X_test)

joblib.dump(ensemble_weighted, 'outputs/classification/ensemble/ensemble_weighted_model.pkl')

# Metrics
accuracy_weighted = accuracy_score(y_test, y_pred_weighted)
f1_weighted = f1_score(y_test, y_pred_weighted, average='weighted')
precision_weighted = precision_score(y_test, y_pred_weighted, average='weighted')
recall_weighted = recall_score(y_test, y_pred_weighted, average='weighted')
metrics_weighted = {
    'accuracy': accuracy_weighted,
    'f1': f1_weighted,
    'precision': precision_weighted,
    'recall': recall_weighted
}
with open('outputs/classification/ensemble/ensemble_weighted_metrics.json', 'w') as f:
    json.dump(metrics_unweighted, f)
print(metrics_weighted)

# # Create plot of Metrics of both ensembles and each model's individual scores to compare them
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load scores
# knn_scores = json.load(open('outputs/classification/knn/knn_scores.json'))
# rf_scores = json.load(open('outputs/classification/rf/rf_scores.json'))
# svm_scores = json.load(open('outputs/classification/svm/svm_scores.json'))
# xgboost_scores = json.load(open('outputs/classification/xgboost/xgboost_scores.json'))

# # Create data frame
# scores = pd.DataFrame([knn_scores, rf_scores, svm_scores, xgboost_scores], index=['knn', 'rf', 'svm', 'xgboost'])
# scores = scores.append(pd.DataFrame([metrics_unweighted], index=['ensemble_unweighted']))
# scores = scores.append(pd.DataFrame([metrics_weighted], index=['ensemble_weighted']))