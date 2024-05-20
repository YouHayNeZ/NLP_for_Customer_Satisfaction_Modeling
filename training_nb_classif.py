# Naive Bayes (Classification)

# Importing the libraries
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from scipy.stats import uniform
from preprocessing import *
from training_helper_func import *

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters (Naive Bayes doesn't have many hyperparameters to tune)
param_dist = { 
    'alpha': uniform(0.0, 1.0),
    'fit_prior': [False, True]
}

# Hyperparameter tuning & CV results
random_search, results = hpo_and_cv_results(MultinomialNB(), 'outputs/classification/nb/nb_cv_results.csv', param_dist, X_train, y_train)

# Parallel coordinate plot
scaler = MinMaxScaler()
results = results.rename(columns={'param_alpha': 'alpha', 'param_fit_prior': 'fit_prior'})
for param in ['alpha', 'fit_prior']:
    results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
results_pc = results.drop(columns=['std_test_score', 'rank_test_score'])
plt.figure(figsize=(14, 7))
parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
plt.legend().remove()
plt.savefig('outputs/classification/nb/nb_parallel_coordinates.png')
plt.show()
# purple = best, yellow = worst

# Count the percentage of each hyperparameter combination (top 10% and bottom 10%)
perc_of_hp_combinations(results, ['fit_prior'])

# Best model and predictions
best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test,
                            'outputs/classification/nb/nb_model.pkl',
                            'outputs/classification/nb/nb_hyperparameters.json',
                            'outputs/classification/nb/nb_train_preds.csv',
                            'outputs/classification/nb/nb_test_preds.csv')

# Confusion matrix and metrics
confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model, 
                            cm_path = 'outputs/classification/nb/nb_confusion_matrix.png', 
                            scores_path = 'outputs/classification/nb/nb_scores.json')

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
    plt.savefig('outputs/classification/nb/roc_curve_class_{}.png'.format(cls))
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
    plt.savefig('outputs/classification/nb/precision_recall_curve_class_{}.png'.format(cls))
    plt.show()

# No feature importance plot for Naive Bayes