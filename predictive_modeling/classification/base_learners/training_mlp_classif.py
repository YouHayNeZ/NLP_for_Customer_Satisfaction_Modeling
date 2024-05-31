# Multi-Layer Perceptron (Classification)

# Importing the libraries
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import loguniform, randint
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    param_dist = {
        'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
        'activation': ['tanh', 'relu', 'logistic'],
        'solver': ['adam', 'sgd'],
        'alpha': loguniform(0.0001, 0.1),
        'learning_rate': ['constant','adaptive'],
        'max_iter': randint(50, 1000),
        'early_stopping': [True]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(MLPClassifier(), 'outputs/predictive_modeling/classification/base_learners/mlp/mlp_cv_results.csv', param_dist, X_train, y_train, n_iter=50)

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_hidden_layer_sizes': 'hidden_layer_sizes', 'param_activation': 'activation', 'param_solver': 'solver', 'param_alpha': 'alpha', 'param_learning_rate': 'learning_rate', 'param_max_iter': 'max_iter', 'param_early_stopping': 'early_stopping'})
    for param in ['alpha', 'max_iter']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['std_test_score', 'rank_test_score', 'hidden_layer_sizes', 'activation', 'solver', 'learning_rate', 'early_stopping'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/classification/base_learners/mlp/mlp_parallel_coordinates.png')
    plt.show()
    # purple = best, yellow = worst

    # Percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['hidden_layer_sizes', 'activation', 'solver', 'learning_rate'])


    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/classification/base_learners/mlp/mlp_model.pkl', 
                            'outputs/predictive_modeling/classification/base_learners/mlp/mlp_hyperparameters.json', 
                            'outputs/predictive_modeling/classification/base_learners/mlp/mlp_train_preds.csv', 
                            'outputs/predictive_modeling/classification/base_learners/mlp/mlp_test_preds.csv')

    # Confusion matrix and metrics
    confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model,
                                    cm_path = 'outputs/predictive_modeling/classification/base_learners/mlp/mlp_confusion_matrix.png',
                                    scores_path = 'outputs/predictive_modeling/classification/base_learners/mlp/mlp_scores.json')

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
        plt.savefig('outputs/predictive_modeling/classification/base_learners/mlp/roc_curve_class_{}.png'.format(cls))
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
        plt.savefig('outputs/predictive_modeling/classification/base_learners/mlp/precision_recall_curve_class_{}.png'.format(cls))
        plt.show()
    
if __name__ == '__main__':
    main()
