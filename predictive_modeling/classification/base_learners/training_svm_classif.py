# Support Vector Machine (Classification)

# Importing the libraries
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler, label_binarize
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    # param_dist = {
    #     'C': uniform(2.5, 1),
    #     'kernel': ['rbf'],
    #     'gamma': ['auto'],
    #     'degree': randint(1, 15),
    #     'class_weight': ['balanced'],
    #     'probability': [True],
    #     'shrinking': [True, False],
    #     'tol': uniform(0.001, 0.2),
    # }

    param_dist = {
        'C': [3.2689177413153976],
        'kernel': ['rbf'],
        'gamma': ['auto'],
        'degree': [4],
        'class_weight': ['balanced'],
        'probability': [True],
        'shrinking': [False],
        'tol': [0.01017225328092659]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(SVC(), 'outputs/predictive_modeling/classification/base_learners/svm/svm_cv_results.csv', param_dist, X_train, y_train, n_iter=1, cv=10)

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_C': 'C', 'param_kernel': 'kernel', 'param_gamma': 'gamma', 'param_degree': 'degree', 'param_class_weight': 'class_weight', 'param_shrinking': 'shrinking', 'param_probability': 'probability', 'param_tol': 'tol'})
    for param in ['C', 'degree', 'tol']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'kernel', 'gamma', 'class_weight', 'probability', 'shrinking'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/classification/base_learners/svm/svm_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Count the percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['kernel', 'gamma', 'shrinking', 'probability', 'class_weight'])

    # Best model and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/classification/base_learners/svm/svm_model.pkl', 
                            'outputs/predictive_modeling/classification/base_learners/svm/svm_hyperparameters.json', 
                            'outputs/predictive_modeling/classification/base_learners/svm/svm_train_preds.csv', 
                            'outputs/predictive_modeling/classification/base_learners/svm/svm_test_preds.csv')

    # Confusion matrix and metrics
    confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model, 
                                cm_path = 'outputs/predictive_modeling/classification/base_learners/svm/svm_confusion_matrix.png', 
                                scores_path = 'outputs/predictive_modeling/classification/base_learners/svm/svm_scores.json')

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
        plt.xlabel('False Positive Rate', fontsize = 16)
        plt.ylabel('True Positive Rate', fontsize = 16)
        plt.legend(loc="lower right", fontsize = 14)
        plt.savefig('outputs/predictive_modeling/classification/base_learners/svm/roc_curve_class_{}.png'.format(class_num))
        plt.close()

        # Create Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test_binary[:, 0], predicted_probabilities[:, 0])

        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall', fontsize = 16)
        plt.ylabel('Precision', fontsize = 16)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig('outputs/predictive_modeling/classification/base_learners/svm/precision_recall_curve_class_{}.png'.format(class_num))
        plt.close()

    # No feature importance for SVM

if __name__ == '__main__':
    main()