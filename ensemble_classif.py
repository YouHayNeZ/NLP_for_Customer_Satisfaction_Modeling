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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import randint, uniform
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import log_loss

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Load models
knn = joblib.load('outputs/classification/knn/knn_model.pkl')
rf = joblib.load('outputs/classification/rf/rf_model.pkl')
svm = joblib.load('outputs/classification/svm/svm_model.pkl')
nb = joblib.load('outputs/classification/nb/nb_model.pkl')




# Create ensemble (unweighted)
ensemble_unweighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('nb', nb)], voting='soft', n_jobs=-1)
ensemble_unweighted.fit(X_train, y_train)
joblib.dump(ensemble_unweighted, 'outputs/classification/ensemble/ensemble_unweighted_model.pkl')

# Predictions
y_pred_unweighted = ensemble_unweighted.predict(X_test) + 1

# Metrics
accuracy_unweighted = accuracy_score(y_test + 1, y_pred_unweighted)
f1_unweighted = f1_score(y_test + 1, y_pred_unweighted, average='weighted')
precision_unweighted = precision_score(y_test + 1, y_pred_unweighted, average='weighted')
recall_unweighted = recall_score(y_test + 1, y_pred_unweighted, average='weighted')
log_loss_unweighted = log_loss(y_test, ensemble_unweighted.predict_proba(X_test))
metrics_unweighted = {
    'accuracy': accuracy_unweighted,
    'f1': f1_unweighted,
    'precision': precision_unweighted,
    'recall': recall_unweighted,
    'logloss': log_loss_unweighted
}
with open('outputs/classification/ensemble/ensemble_unweighted_metrics.json', 'w') as f:
    json.dump(metrics_unweighted, f)
print(metrics_unweighted)






# Create ensemble (weighted)
log_loss_scores = []

while len(log_loss_scores) < 30:
    w_rf = random.uniform(0, 1)
    w_knn = random.uniform(0, 1)
    w_svm = random.uniform(0, 1)
    w_nb = random.uniform(0, 1)
    ensemble_weighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('nb', nb)], voting='soft', n_jobs=-1, weights=[w_knn, w_rf, w_svm, w_nb])
    ensemble_weighted.fit(X_train, y_train)
    y_pred = ensemble_weighted.predict(X_test)
    log_loss_score = log_loss(y_test, ensemble_weighted.predict_proba(X_test))
    log_loss_scores.append([w_knn, w_rf, w_svm, w_nb, log_loss_score])

# Convert the list to a DataFrame
log_loss_scores_df = pd.DataFrame(log_loss_scores, columns=['w_knn', 'w_rf', 'w_svm', 'w_nb', 'log_loss'])

# Use parallel coordinate plot for weights and log loss scores
plt.figure(figsize=(14, 7))
parallel_coordinates(log_loss_scores_df, 'log_loss', colormap='viridis', alpha=0.25)
plt.legend().remove()
plt.savefig('outputs/classification/ensemble/ensemble_parallel_coordinates.png')
plt.show()

log_loss_scores = log_loss_scores_df.sort_values(by='log_loss', ascending=True)
log_loss_scores.to_csv('outputs/classification/ensemble/ensemble_weighted_log_loss_scores.csv')
best_weights = log_loss_scores.iloc[0, :-1].values

ensemble_weighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('nb', nb)], voting='soft', n_jobs=-1, weights=best_weights)
ensemble_weighted.fit(X_train, y_train)
y_pred_weighted = ensemble_weighted.predict(X_test) + 1

joblib.dump(ensemble_weighted, 'outputs/classification/ensemble/ensemble_weighted_model.pkl')

# Metrics
accuracy_weighted = accuracy_score(y_test + 1, y_pred_weighted)
f1_weighted = f1_score(y_test + 1, y_pred_weighted, average='weighted')
precision_weighted = precision_score(y_test + 1, y_pred_weighted, average='weighted')
recall_weighted = recall_score(y_test + 1, y_pred_weighted, average='weighted')
log_loss_weighted = log_loss(y_test, ensemble_weighted.predict_proba(X_test))
metrics_weighted = {
    'accuracy': accuracy_weighted,
    'f1': f1_weighted,
    'precision': precision_weighted,
    'recall': recall_weighted,
    'logloss': log_loss_weighted
}
with open('outputs/classification/ensemble/ensemble_weighted_metrics.json', 'w') as f:
    json.dump(metrics_weighted, f)
print(metrics_weighted)







# Stacking models (using XGBoost)

# Create data frame with training predictions of all models and real labels
X_train_stacking = pd.DataFrame({
    'knn': knn.predict(X_train) + 1,
    'rf': rf.predict(X_train) + 1,
    'svm': svm.predict(X_train) + 1,
    'nb': nb.predict(X_train) + 1
})

X_val_stacking = pd.DataFrame({
    'knn': knn.predict(X_val) + 1,
    'rf': rf.predict(X_val) + 1,
    'svm': svm.predict(X_val) + 1,
    'nb': nb.predict(X_val) + 1
})

X_test_stacking = pd.DataFrame({
    'knn': knn.predict(X_test) + 1,
    'rf': rf.predict(X_test) + 1,
    'svm': svm.predict(X_test) + 1,
    'nb': nb.predict(X_test) + 1
})

# Concatenate all 3 data frames
X_stacked = pd.concat([X_train_stacking, X_val_stacking, X_test_stacking])
X_stacked = X_stacked.reset_index(drop=True)

# Load original data
data = pd.read_csv('data/ryanair_reviews.csv')
data = data.dropna(subset=['Overall Rating'])
data = pd.concat([data, X_stacked], axis=1)
data.to_csv('outputs/classification/ensemble/stacking_data.csv', index=False)

# Preprocess stacking data
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('outputs/classification/ensemble/stacking_data.csv')

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
random_search = RandomizedSearchCV(estimator=xgb.XGBClassifier(), 
                               param_distributions=param_dist, 
                               n_iter=50, 
                               cv=2, 
                               verbose=2, 
                               scoring='neg_log_loss',
                               random_state=42, 
                               n_jobs=-1)
random_search.fit(X_train, y_train,
                    early_stopping_rounds=10,
                    eval_set=[(X_val, y_val)],
                    eval_metric='mlogloss',
                  )

# Save results of RandomizedSearchCV
results = pd.DataFrame(random_search.cv_results_)
interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
results = results[interested_columns]
results = results.sort_values(by='rank_test_score')
results['mean_test_score'] = results['mean_test_score']
results.to_csv('outputs/classification/ensemble/xgboost_ensemble_cv_results.csv')

# Parallel coordinate plot without max_features and bootstrap
scaler = MinMaxScaler()
results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_min_child_weight': 'min_child_weight', 'param_colsample_bytree': 'colsample_bytree', 'param_colsample_bylevel': 'colsample_bylevel', 'param_reg_lambda': 'reg_lambda', 'param_reg_alpha': 'reg_alpha', 'param_scale_pos_weight': 'scale_pos_weight', 'param_gamma': 'gamma'})
for param in ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'colsample_bytree', 'colsample_bylevel', 'reg_lambda', 'reg_alpha', 'scale_pos_weight', 'gamma']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results = results.drop(columns=['std_test_score', 'rank_test_score'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha = 0.25)
plt.legend().remove()
plt.savefig('outputs/classification/ensemble/xgboost_ensemble_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/classification/ensemble/xgboost_ensemble_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/classification/ensemble/xgboost_ensemble_hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)

# Make predictions, save them as data frame and set flown date as index
X_train = pd.concat([X_train, X_val])
y_train = pd.concat([y_train, y_val])
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Correct classes again: add +1 to predictions & real values to get the real rating
train_preds = train_preds + 1
test_preds = test_preds + 1
y_train = y_train + 1
y_test = y_test + 1

train_preds = pd.DataFrame({'Predicted Overall Rating': train_preds, 'Real Overall Rating': y_train, 'Date Flown': datetime_train['Date Flown']}).set_index('Date Flown')
test_preds = pd.DataFrame({'Predicted Overall Rating': test_preds, 'Real Overall Rating': y_test, 'Date Flown': datetime_test['Date Flown']}).set_index('Date Flown')

train_preds.to_csv('outputs/classification/ensemble/xgboost_ensemble_train_preds.csv')
test_preds.to_csv('outputs/classification/ensemble/xgboost_ensemble_test_preds.csv')

# Making the Confusion Matrix
predicted_labels = test_preds['Predicted Overall Rating']

cm = confusion_matrix(y_test, predicted_labels)
accuracy = accuracy_score(y_test, predicted_labels)
classification_report(y_test, predicted_labels)
f1_score = f1_score(y_test, predicted_labels, average='weighted')
precision = precision_score(y_test, predicted_labels, average='weighted')
recall = recall_score(y_test, predicted_labels, average='weighted')
log_loss_xgboost = log_loss(y_test, best_model.predict_proba(X_test))

# Save scores
scores = {
    'accuracy': accuracy,
    'f1_score': f1_score,
    'precision': precision,
    'recall': recall,
    'logloss': log_loss_xgboost
}
print(scores)
with open('outputs/classification/ensemble/xgboost_ensemble_scores.json', 'w') as f:
    json.dump(scores, f)

# Create ROC & PR curves for classes 1, 5, and 10
probabilities = best_model.predict_proba(X_test)
classes = [1, 5, 10]

for cls in classes:
    # Create ROC curve
    fpr, tpr, _ = roc_curve((y_test == cls).astype(int), probabilities[:, cls - 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Class {})'.format(cls))
    plt.legend(loc="lower right")
    plt.savefig('outputs/classification/ensemble/roc_curve_class_{}.png'.format(cls))
    plt.show()

    # Create Precision-Recall curve
    precision, recall, _ = precision_recall_curve((y_test == cls).astype(int), probabilities[:, cls - 1])

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve (Class {})'.format(cls))
    plt.savefig('outputs/classification/ensemble/precision_recall_curve_class_{}.png'.format(cls))
    plt.show()

# Create feature importance plot for top 15
feature_importance = best_model.feature_importances_
features = X_train.columns
feature_importance_scores = dict(zip(features, feature_importance))
feature_importance_df = pd.DataFrame(feature_importance_scores.items(), columns=['Feature', 'Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.to_csv('outputs/classification/ensemble/xgboost_ensemble_feature_importance.csv', index=False)

feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
plt.figure(figsize=(14, 7))
plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
plt.ylabel('Feature Importance')
plt.xlabel('Feature')
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.savefig('outputs/classification/ensemble/xgboost_ensemble_feature_importance.png')
plt.show()

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/classification/ensemble/xgboost_ensemble_train_predictions.png')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(14, 7))
plt.matshow(cm, cmap='viridis')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.title('Confusion Matrix')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
plt.savefig('outputs/classification/ensemble/xgboost_ensemble_confusion_matrix.png')
plt.show()