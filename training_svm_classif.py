# Support Vector Machine (Classification)

# Importing the libraries
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler, label_binarize
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
from preprocessing import *
from training_helper_func import *

# Prepare data for training
X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews.csv')

# Define the range of hyperparameters
param_dist = {
    'C': uniform(0.000001, 100),
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto'],
    'degree': randint(1, 10),
    'class_weight': ['balanced'],
    'probability': [True]
}

# Hyperparameter tuning & CV results
random_search, results = hpo_and_cv_results(SVC(), 'outputs/classification/svm/svm_cv_results.csv', param_dist, X_train, y_train)

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

# Count the percentage of each hyperparameter combination (top 10% and bottom 10%)
perc_of_hp_combinations(results, ['kernel', 'gamma'])

# Best model and predictions
best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                           'outputs/classification/svm/svm_model.pkl', 
                           'outputs/classification/svm/svm_hyperparameters.json', 
                           'outputs/classification/svm/svm_train_preds.csv', 
                           'outputs/classification/svm/svm_test_preds.csv')

# Confusion matrix and metrics
confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model, 
                             cm_path = 'outputs/classification/svm/svm_confusion_matrix.png', 
                             scores_path = 'outputs/classification/svm/svm_scores.json')

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
