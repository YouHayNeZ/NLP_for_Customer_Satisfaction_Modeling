# XGBoost (Classification)

# Importing the libraries
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from preprocessing import *
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json

# Importing the dataset
data = create_pipeline('data/ryanair_reviews.csv')

# Splitting the dataset into the Training set and Test set
X = data.drop(columns=['Overall Rating'])
y = data['Overall Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
y_train = y_train.astype(int) - 1
y_test = y_test.astype(int) - 1

# Extract dates from train and test sets
datetime_train = X_train[['Date Flown']]
datetime_test = X_test[['Date Flown']]

# Remove dates from train and test sets
X_train = X_train.drop(columns=['Date Published', 'Date Flown'])
X_test = X_test.drop(columns=['Date Published', 'Date Flown'])

# Remove 'Comment title' and 'Comment' columns
X_train = X_train.drop(columns=['Comment title', 'Comment'])
X_test = X_test.drop(columns=['Comment title', 'Comment'])


# Define the range of hyperparameters
param_dist = {
    'n_estimators': randint(100, 2000),
    'learning_rate': uniform(0.0001, 0.1),  
    'max_depth': randint(1, 20),  
    'min_child_weight': randint(1, 21),  
    'subsample': uniform(0.0, 1.0),  
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
                               n_iter=100, 
                               cv=5, 
                               verbose=2, 
                               scoring='f1_weighted',
                               random_state=42, 
                               n_jobs=-1)
random_search.fit(X_train, y_train)

# Save results of RandomizedSearchCV
results = pd.DataFrame(random_search.cv_results_)
interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
results = results[interested_columns]
results = results.sort_values(by='rank_test_score')
results['mean_test_score'] = results['mean_test_score']
results.to_csv('outputs/classification/xgboost/xgboost_cv_results.csv')

# Parallel coordinate plot without max_features and bootstrap
scaler = MinMaxScaler()
results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_min_child_weight': 'min_child_weight', 'param_subsample': 'subsample', 'param_colsample_bytree': 'colsample_bytree', 'param_colsample_bylevel': 'colsample_bylevel', 'param_reg_lambda': 'reg_lambda', 'param_reg_alpha': 'reg_alpha', 'param_scale_pos_weight': 'scale_pos_weight', 'param_gamma': 'gamma'})
for param in ['n_estimators', 'learning_rate', 'max_depth', 'min_child_weight', 'subsample', 'colsample_bytree', 'colsample_bylevel', 'reg_lambda', 'reg_alpha', 'scale_pos_weight', 'gamma']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results = results.drop(columns=['std_test_score', 'rank_test_score'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha = 0.25)
plt.legend().remove()
plt.savefig('outputs/classification/xgboost/xgboost_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/classification/xgboost/xgboost_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/classification/xgboost/xgboost_hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)

# Make predictions, save them as data frame and set flown date as index
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

# Correct classes again: add +1 to predictions & real values to get the real rating
train_preds = train_preds + 1
test_preds = test_preds + 1
y_train = y_train + 1
y_test = y_test + 1

train_preds = pd.DataFrame({'Predicted Overall Rating': train_preds, 'Real Overall Rating': y_train, 'Date Flown': datetime_train['Date Flown']}).set_index('Date Flown')
test_preds = pd.DataFrame({'Predicted Overall Rating': test_preds, 'Real Overall Rating': y_test, 'Date Flown': datetime_test['Date Flown']}).set_index('Date Flown')

train_preds.to_csv('outputs/classification/xgboost/xgboost_train_preds.csv')
test_preds.to_csv('outputs/classification/xgboost/xgboost_test_preds.csv')

# Making the Confusion Matrix
predicted_labels = test_preds['Predicted Overall Rating']

cm = confusion_matrix(y_test, predicted_labels)
accuracy = accuracy_score(y_test, predicted_labels)
classification_report(y_test, predicted_labels)
f1_score = f1_score(y_test, predicted_labels, average='weighted')
precision = precision_score(y_test, predicted_labels, average='weighted')
recall = recall_score(y_test, predicted_labels, average='weighted')

# Save scores
scores = {
    'accuracy': accuracy,
    'f1_score': f1_score,
    'precision': precision,
    'recall': recall
}
print(scores)
with open('outputs/classification/xgboost/xgboost_scores.json', 'w') as f:
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
    plt.savefig('outputs/classification/xgboost/roc_curve_class_{}.png'.format(cls))
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
    plt.savefig('outputs/classification/xgboost/precision_recall_curve_class_{}.png'.format(cls))
    plt.show()

# Create feature importance plot for top 15
feature_importance = best_model.feature_importances_
features = X_train.columns
feature_importance_scores = dict(zip(features, feature_importance))
feature_importance_df = pd.DataFrame(feature_importance_scores.items(), columns=['Feature', 'Importance'])
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
feature_importance_df.to_csv('outputs/classification/xgboost/xgboost_feature_importance.csv', index=False)

feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
plt.figure(figsize=(14, 7))
plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
plt.ylabel('Feature Importance')
plt.xlabel('Feature')
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.savefig('outputs/classification/xgboost/xgboost_feature_importance.png')
plt.show()

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/classification/xgboost/xgboost_train_predictions.png')
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
plt.savefig('outputs/classification/xgboost/xgboost_confusion_matrix.png')
plt.show()