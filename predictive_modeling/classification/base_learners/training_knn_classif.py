# K-Nearest Neighbors (Classification)

# Importing the libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from scipy.stats import uniform, randint, rv_discrete
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define a custom distribution for n_neighbors
    #n_neighbors_choices = [randint(5, 21), randint(90, 111), randint(140, 181)]

    # Define the range of hyperparameters
    # param_dist = {
    #     #'n_neighbors': np.random.choice(n_neighbors_choices),
    #     'n_neighbors': [98],
    #     'weights': ['distance'],
    #     'algorithm': ['kd_tree'],
    #     'leaf_size': randint(360, 600),
    #     'p': uniform(1, 400),
    #     'metric': ['manhattan']
    # }

    param_dist = {
        'n_neighbors': [98],
        'weights': ['distance'],
        'algorithm': ['kd_tree'],
        'leaf_size': [462],
        'p': [319.6171947440931],
        'metric': ['manhattan']
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(KNeighborsClassifier(), 'outputs/predictive_modeling/classification/base_learners/knn/knn_cv_results.csv', param_dist, X_train, y_train, n_iter=1, cv=10)

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_neighbors': 'n_neighbors', 'param_weights': 'weights', 'param_algorithm': 'algorithm', 'param_leaf_size': 'leaf_size', 'param_p': 'p', 'param_metric': 'metric'})
    for param in ['n_neighbors', 'leaf_size', 'p']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'algorithm', 'weights', 'metric'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/classification/base_learners/knn/knn_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Count the percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['algorithm', 'weights', 'metric'])

    # Best model and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test,
                                'outputs/predictive_modeling/classification/base_learners/knn/knn_model.pkl',
                                'outputs/predictive_modeling/classification/base_learners/knn/knn_hyperparameters.json',
                                'outputs/predictive_modeling/classification/base_learners/knn/knn_train_preds.csv',
                                'outputs/predictive_modeling/classification/base_learners/knn/knn_test_preds.csv')

    # Confusion matrix and metrics
    confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model,
                                    cm_path = 'outputs/predictive_modeling/classification/base_learners/knn/knn_confusion_matrix.png',
                                    scores_path = 'outputs/predictive_modeling/classification/base_learners/knn/knn_scores.json')

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
        plt.xlabel('False Positive Rate', fontsize=16)
        plt.ylabel('True Positive Rate', fontsize=16)
        plt.legend(loc="lower right", fontsize=14)
        plt.savefig('outputs/predictive_modeling/classification/base_learners/knn/roc_curve_class_{}.png'.format(cls))
        plt.close()

        # Create Precision-Recall curve
        precision, recall, _ = precision_recall_curve((y_test == cls).astype(int), probabilities[:, cls - 1])

        plt.figure()
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall', fontsize = 16)
        plt.ylabel('Precision', fontsize = 16)
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.savefig('outputs/predictive_modeling/classification/base_learners/knn/precision_recall_curve_class_{}.png'.format(cls))
        plt.close()

if __name__ == '__main__':
    main()