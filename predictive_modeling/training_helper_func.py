# Helper functions for model training

# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score, mean_absolute_error, mean_squared_error, r2_score, log_loss
from sklearn.model_selection import RandomizedSearchCV
import joblib
import matplotlib.pyplot as plt
import json

# Hyperparameter tuning and CV results
def hpo_and_cv_results(estimator, cv_results_path, param_dist, X_train, y_train, scoring='neg_log_loss', n_iter=1500, cv=5, verbose=2, random_state=42, n_jobs=-1):
    random_search = RandomizedSearchCV(estimator=estimator, 
                                       param_distributions=param_dist, 
                                       n_iter=n_iter, 
                                       cv=cv, 
                                       verbose=verbose, 
                                       scoring=scoring,
                                       random_state=random_state, 
                                       n_jobs=n_jobs)
    random_search.fit(X_train, y_train)

    results = pd.DataFrame(random_search.cv_results_)
    interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[interested_columns]
    results = results.sort_values(by='rank_test_score')
    results['mean_test_score'] = results['mean_test_score']
    results.to_csv(cv_results_path)

    return random_search, results

# Count the percentage of each hyperparameter combination
def perc_of_hp_combinations(results, list, top_percentage=0.1):
    results = results.sort_values(by='mean_test_score', ascending=False)
    results_top10 = results.head(int(len(results) * top_percentage))
    results_bottom10 = results.tail(int(len(results) * top_percentage))

    print('Out of top 10% of models (number {}):'.format(len(results_top10)))
    for param in list:
        for value in results_top10[param].unique():
            print(f'{param} = {value}: {len(results_top10[results_top10[param] == value])/len(results_top10)}')

    print('Out of bottom 10% of models (number {}):'.format(len(results_bottom10)))
    for param in list:
        for value in results_bottom10[param].unique():
            print(f'{param} = {value}: {len(results_bottom10[results_bottom10[param] == value])/len(results_bottom10)}')

# Best model, hyperparameters, and predictions
def best_model_and_predictions(estimator, X_train, X_test, y_train, y_test, datetime_train, datetime_test, model_path, hyperparameters_path, train_preds_path, test_preds_path):
    best_model = estimator.best_estimator_
    joblib.dump(best_model, model_path)

    hyperparameters = estimator.best_params_
    with open(hyperparameters_path, 'w') as f:
        json.dump(hyperparameters, f)

    train_preds = best_model.predict(X_train)
    test_preds = best_model.predict(X_test)

    train_preds = np.round(train_preds + 1)
    test_preds = np.round(test_preds + 1)
    y_train = y_train + 1
    y_test = y_test + 1

    train_preds = pd.DataFrame({'Predicted Overall Rating': train_preds, 'Real Overall Rating': y_train, 'Date Flown': datetime_train['Date Flown']}).set_index('Date Flown')
    test_preds = pd.DataFrame({'Predicted Overall Rating': test_preds, 'Real Overall Rating': y_test, 'Date Flown': datetime_test['Date Flown']}).set_index('Date Flown')

    train_preds.to_csv(train_preds_path)
    test_preds.to_csv(test_preds_path)

    return best_model, train_preds, test_preds, y_train, y_test

# Confusion matrix and metrics (only classification)
def confusion_matrix_and_metrics(X_test, y_test, test_preds, model, cm_path, scores_path):
    predicted_labels = test_preds['Predicted Overall Rating']

    logloss = log_loss(y_test, model.predict_proba(X_test))
    cm = confusion_matrix(y_test, predicted_labels)
    accuracy = accuracy_score(y_test, predicted_labels)
    classification_report(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels, average='weighted')
    precision = precision_score(y_test, predicted_labels, average='weighted')
    recall = recall_score(y_test, predicted_labels, average='weighted')

    scores = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'logloss': logloss
    }
    print(scores)
    with open(scores_path, 'w') as f:
        json.dump(scores, f)

    plt.figure(figsize=(14, 7))
    plt.matshow(cm, cmap='viridis')
    plt.colorbar()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white')
    plt.savefig(cm_path)
    plt.show()

# Regression metrics (only regression)
def regression_metrics(y_test, test_preds, scores_path):
    mae = mean_absolute_error(y_test, test_preds['Predicted Overall Rating'])
    mse = mean_squared_error(y_test, test_preds['Predicted Overall Rating'])
    r2 = r2_score(y_test, test_preds['Predicted Overall Rating'])

    scores = {
        'mae': mae,
        'mse': mse,
        'r2': r2
    }
    print(scores)
    with open(scores_path, 'w') as f:
        json.dump(scores, f)

# Plot predictions vs real values over time (only regression)
def test_preds_vs_real_over_time(test_preds, path):
    test_predictions = test_preds.groupby('Date Flown').mean()

    plt.figure(figsize=(14, 7))
    plt.plot(test_predictions.index, test_predictions['Predicted Overall Rating'], label='Predicted Overall Rating (Test)')
    plt.plot(test_predictions.index, test_predictions['Real Overall Rating'], label='Real Overall Rating (Test)')
    plt.legend()
    plt.title('Predicted vs Real Overall Rating over Time')
    plt.savefig(path)
    plt.show()