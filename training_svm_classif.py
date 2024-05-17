# Support Vector Machine (Classification)

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from preprocessing import *
from scipy.stats import randint, uniform
import joblib
from sklearn.preprocessing import MinMaxScaler, label_binarize
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json
from sklearn.metrics import log_loss

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters
param_dist = {
    'C': uniform(0.000001, 100),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': randint(1, 10),
    'class_weight': ['balanced']
}

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=SVC(), 
                                   param_distributions=param_dist, 
                                   n_iter=1500, 
                                   cv=5, 
                                   verbose=2, 
                                   scoring='neg_log_loss',
                                   random_state=42, 
                                   n_jobs=-1)
random_search.fit(X_train, y_train)

# Save results of RandomizedSearchCV
results = pd.DataFrame(random_search.cv_results_)
interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
results = results[interested_columns]
results = results.sort_values(by='rank_test_score')
results['mean_test_score'] = results['mean_test_score']
results.to_csv('outputs/classification/svm/svm_cv_results.csv')

# Parallel coordinate plot
scaler = MinMaxScaler()
results = results.rename(columns={'param_C': 'C', 'param_kernel': 'kernel', 'param_gamma': 'gamma', 'param_degree': 'degree', 'param_class_weight': 'class_weight'})
for param in ['C', 'degree']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'kernel', 'gamma', 'class_weight'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
plt.legend().remove()
plt.savefig('outputs/classification/svm/svm_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Count the percentage of each hyperparameter combination
results = results.sort_values(by='mean_test_score', ascending=False)
results_top10 = results.head(int(len(results) * 0.1))
results_bottom10 = results.tail(int(len(results) * 0.1))

# Print length of top 10% and bottom 10% of models
print('Out of top 10% of models (number {}):'.format(len(results_top10)))
for param in ['kernel', 'gamma']:
    for value in results_top10[param].unique():
        print(f'{param} = {value}: {len(results_top10[results_top10[param] == value])/len(results_top10)}')

print('Out of bottom 10% of models (number {}):'.format(len(results_bottom10)))
for param in ['kernel', 'gamma']:
    for value in results_bottom10[param].unique():
        print(f'{param} = {value}: {len(results_bottom10[results_bottom10[param] == value])/len(results_bottom10)}')

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/classification/svm/svm_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/classification/svm/svm_hyperparameters.json', 'w') as f:
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

train_preds.to_csv('outputs/classification/svm/svm_train_preds.csv')
test_preds.to_csv('outputs/classification/svm/svm_test_preds.csv')

# Making the Confusion Matrix
predicted_labels = test_preds['Predicted Overall Rating']

logloss = log_loss(y_test, best_model.predict_proba(X_test))
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
    'recall': recall,
    'logloss': logloss
}
print(scores)
with open('outputs/classification/svm/svm_scores.json', 'w') as f:
    json.dump(scores, f)

# Create ROC & PR curves for classes 1, 5, and 10
for class_num in [1, 5, 10]:
    y_test_binary = label_binarize(y_test, classes=[class_num])
    predicted_probabilities = best_model.decision_function(X_test)

    # Create ROC curve
    fpr, tpr, _ = roc_curve(y_test_binary[:, 0], predicted_probabilities[:, 0])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Class {})'.format(class_num))
    plt.legend(loc="lower right")
    plt.savefig('outputs/classification/svm/roc_curve_class_{}.png'.format(class_num))
    plt.show()

    # Create Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test_binary[:, 0], predicted_probabilities[:, 0])

    plt.figure()
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve (Class {})'.format(class_num))
    plt.savefig('outputs/classification/svm/precision_recall_curve_class_{}.png'.format(class_num))
    plt.show()

# No feature importance for SVM (maybe add coefficient based feature importance later)

# Plot predictions vs real values over time (use average rating per 'Date Flown' to make it more readable)
train_predictions = train_preds.groupby('Date Flown').mean()
test_predictions = test_preds.groupby('Date Flown').mean()

plt.figure(figsize=(14, 7))
plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
plt.legend()
plt.title('Predicted vs Real Overall Rating over Time')
plt.savefig('outputs/classification/rf/rf_train_predictions.png')
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
plt.savefig('outputs/classification/svm/svm_confusion_matrix.png')
plt.show()
