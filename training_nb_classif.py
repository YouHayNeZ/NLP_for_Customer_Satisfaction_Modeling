# Naive Bayes (Classification)



#### NOT WORKING YET, SOME DATA ERROR IS OCURRING!





# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_curve, auc, precision_recall_curve
from preprocessing import *
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
import json
from scipy.stats import uniform

# Importing the dataset
data = create_pipeline('data/ryanair_reviews.csv')

# Splitting the dataset into the Training set and Test set
X = data.drop(columns=['Overall Rating'])
y = data['Overall Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Extract dates from train and test sets
datetime_train = X_train[['Date Flown']]
datetime_test = X_test[['Date Flown']]

# Remove dates from train and test sets
X_train = X_train.drop(columns=['Date Published', 'Date Flown'])
X_test = X_test.drop(columns=['Date Published', 'Date Flown'])

# Remove 'Comment title' and 'Comment' columns
X_train = X_train.drop(columns=['Comment title', 'Comment'])
X_test = X_test.drop(columns=['Comment title', 'Comment'])

# Define the range of hyperparameters (Naive Bayes doesn't have many hyperparameters to tune)
param_dist = { 
    'alpha': uniform(0.0, 1.0),
    'fit_prior': [True, False]
}

# Hyperparameter tuning
random_search = RandomizedSearchCV(estimator=CategoricalNB(), 
                                   param_distributions=param_dist, 
                                   n_iter=10,
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
results.to_csv('outputs/classification/rf/rf_cv_results.csv')

# Scatterplot of alpha and fit_prior (colored by mean_test_score)
plt.figure(figsize=(14, 7))
plt.scatter(results['param_alpha'], results['param_fit_prior'], c=results['mean_test_score'], cmap='viridis')
plt.colorbar()
plt.xlabel('alpha')
plt.ylabel('fit_prior')
plt.title('Hyperparameter Surface')
plt.savefig('outputs/classification/nb/nb_randomized_search.png')
plt.show()
# purple = best, yellow = worst

# Get the best model
best_model = random_search.best_estimator_
joblib.dump(best_model, 'outputs/classification/nb/nb_model.pkl')

hyperparameters = random_search.best_params_
with open('outputs/classification/nb/nb_hyperparameters.json', 'w') as f:
    json.dump(hyperparameters, f)

# Make predictions, save them as data frame
train_preds = best_model.predict(X_train)
test_preds = best_model.predict(X_test)

train_preds = pd.DataFrame({'Predicted Overall Rating': train_preds, 'Real Overall Rating': y_train})
test_preds = pd.DataFrame({'Predicted Overall Rating': test_preds, 'Real Overall Rating': y_test})

train_preds.to_csv('outputs/classification/nb/nb_train_preds.csv')
test_preds.to_csv('outputs/classification/nb/nb_test_preds.csv')

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
with open('outputs/classification/nb/nb_scores.json', 'w') as f:
    json.dump(scores, f)

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
plt.savefig('outputs/classification/nb/nb_confusion_matrix.png')
plt.show()
