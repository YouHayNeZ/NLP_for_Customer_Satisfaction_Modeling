# Random Forest (Classification)

# Importing the libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
import matplotlib.pyplot as plt
from scipy.stats import randint
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    feature_selection = True
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv', feature_selection=feature_selection)

    # Define the range of hyperparameters
    param_dist = {
        'n_estimators': randint(500, 1900),
        'max_features': ["sqrt"],
        'max_depth': randint(50, 500),
        'min_samples_split': randint(20, 35),
        'min_samples_leaf': randint(1, 5),
        'bootstrap': [False]
    }

    # Hyperparameter tuning & CV results
    random_search, results = hpo_and_cv_results(RandomForestClassifier(criterion='log_loss'), 'outputs/predictive_modeling/classification/base_learners/rf/rf_cv_results.csv', param_dist, X_train, y_train, n_iter=500)

    # Parallel coordinate plot without max_features and bootstrap
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_max_features': 'max_features', 'param_max_depth': 'max_depth', 'param_min_samples_split': 'min_samples_split', 'param_min_samples_leaf': 'min_samples_leaf', 'param_bootstrap': 'bootstrap'})
    for param in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results_pc = results.drop(columns=['max_features', 'bootstrap', 'std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results_pc, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.savefig('outputs/predictive_modeling/classification/base_learners/rf/rf_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Count the percentage of each hyperparameter combination (top 10% and bottom 10%)
    perc_of_hp_combinations(results, ['bootstrap', 'max_features'])

    # Best model and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test,
                                'outputs/predictive_modeling/classification/base_learners/rf/rf_model.pkl',
                                'outputs/predictive_modeling/classification/base_learners/rf/rf_hyperparameters.json',
                                'outputs/predictive_modeling/classification/base_learners/rf/rf_train_preds.csv',
                                'outputs/predictive_modeling/classification/base_learners/rf/rf_test_preds.csv')

    # Confusion matrix and metrics
    confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model, 
                                    cm_path = 'outputs/predictive_modeling/classification/base_learners/rf/rf_confusion_matrix.png', 
                                    scores_path = 'outputs/predictive_modeling/classification/base_learners/rf/rf_scores.json')

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
        plt.savefig('outputs/predictive_modeling/classification/base_learners/rf/roc_curve_class_{}.png'.format(cls))
        plt.close()

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
        plt.savefig('outputs/predictive_modeling/classification/base_learners/rf/precision_recall_curve_class_{}.png'.format(cls))
        plt.close()

    # Save feature importance scores
    feature_importance = best_model.feature_importances_
    features = X_train.columns
    feature_importance_scores = dict(zip(features, feature_importance))
    feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True))
    feature_importance_df = pd.DataFrame.from_dict(feature_importance_scores, orient='index', columns=['importance'])
    feature_importance_df.index.name = 'feature'
    if feature_selection:
        feature_importance_df.to_csv('outputs/predictive_modeling/classification/base_learners/rf/rf_feature_importance.csv')
    else:
        feature_importance_df.to_csv('outputs/predictive_modeling/feature_selection/feature_importance_scores.csv')

    # Feature imporance plot (with visible feature names, only top 15)
    feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.title('Feature Importance Plot (Top 15)')
    plt.savefig('outputs/predictive_modeling/classification/base_learners/rf/rf_feature_importance_top_15.png')
    plt.close()
    
    if feature_selection==False:
        # Feature importance plot (without feature names, all features)
        feature_importance = pd.read_csv('outputs/predictive_modeling/classification/feature_selection/feature_importance_scores.csv')['importance']
        plt.figure(figsize=(14, 7))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.axvline(x=100, color='r', linestyle='--')
        plt.text(150, 0.05, 'Cutoff Threshold at 100 \n Parameters', verticalalignment='center', horizontalalignment='center', size=15, color='r')
        plt.ylabel('Feature Importance')
        plt.xlabel('Feature')
        plt.title('Feature Importance (All Features)')
        plt.savefig('outputs/predictive_modeling/classification/feature_selection/feature_importance_image.png')
        plt.close()

if __name__ == '__main__':
    main()