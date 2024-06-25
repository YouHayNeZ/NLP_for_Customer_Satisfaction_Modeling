import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, auc, log_loss, precision_recall_curve
from sklearn.ensemble import VotingClassifier
import lightgbm as lgb
import json
import random
import matplotlib.pyplot as plt
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing import *
from predictive_modeling.training_helper_func import *

def main():
    # Prepare data for training
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Load models
    knn = joblib.load('outputs/predictive_modeling/classification/base_learners/knn/knn_model.pkl')
    rf = joblib.load('outputs/predictive_modeling/classification/base_learners/rf/rf_model.pkl')
    svm = joblib.load('outputs/predictive_modeling/classification/base_learners/svm/svm_model.pkl')
    nb = joblib.load('outputs/predictive_modeling/classification/base_learners/nb/nb_model.pkl')
    mlp = joblib.load('outputs/predictive_modeling/classification/base_learners/mlp/mlp_model.pkl')




    # # Create ensemble: majority voting (unweighted)
    # ensemble_unweighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('nb', nb), ('mlp', mlp)], voting='soft', n_jobs=-1)
    # ensemble_unweighted.fit(X_train, y_train)
    # joblib.dump(ensemble_unweighted, 'outputs/predictive_modeling/classification/ensemble/ensemble_unweighted_model.pkl')

    # # Predictions
    # y_pred_unweighted = ensemble_unweighted.predict(X_test) + 1

    # # Metrics
    # accuracy_unweighted = accuracy_score(y_test + 1, y_pred_unweighted)
    # f1_unweighted = f1_score(y_test + 1, y_pred_unweighted, average='weighted')
    # precision_unweighted = precision_score(y_test + 1, y_pred_unweighted, average='weighted')
    # recall_unweighted = recall_score(y_test + 1, y_pred_unweighted, average='weighted')
    # log_loss_unweighted = log_loss(y_test, ensemble_unweighted.predict_proba(X_test))
    # metrics_unweighted = {
    #     'accuracy': accuracy_unweighted,
    #     'f1': f1_unweighted,
    #     'precision': precision_unweighted,
    #     'recall': recall_unweighted,
    #     'logloss': log_loss_unweighted
    # }
    # with open('outputs/predictive_modeling/classification/ensemble/ensemble_unweighted_metrics.json', 'w') as f:
    #     json.dump(metrics_unweighted, f)
    # print(metrics_unweighted)






    # # Create ensemble: majority voting (weighted)
    # log_loss_scores = []
    # weights = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # while len(log_loss_scores) < 1000:
    #     w_rf = random.uniform(0, 1)
    #     w_knn = random.uniform(0, 1)
    #     w_svm = random.uniform(0, 1)
    #     w_nb = random.uniform(0, 1)
    #     w_mlp = random.uniform(0, 1)
    #     ensemble_weighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('nb', nb), ('mlp', mlp)], voting='soft', n_jobs=-1, weights=[w_knn, w_rf, w_svm, w_nb, w_mlp])
    #     ensemble_weighted.fit(X_train, y_train)
    #     log_loss_score = log_loss(y_val, ensemble_weighted.predict_proba(X_val))
    #     log_loss_scores.append([w_knn, w_rf, w_svm, w_nb, w_mlp, log_loss_score])
    #     print("finished iteration ", len(log_loss_scores))

    # # Convert the list to a DataFrame
    # log_loss_scores_df = pd.DataFrame(log_loss_scores, columns=['w_knn', 'w_rf', 'w_svm', 'w_nb', 'w_mlp', 'log_loss'])

    # # Use parallel coordinate plot for weights and log loss scores
    # plt.figure(figsize=(14, 7))
    # parallel_coordinates(log_loss_scores_df, 'log_loss', colormap='viridis', alpha=0.25)
    # plt.legend().remove()
    # plt.savefig('outputs/predictive_modeling/classification/ensemble/ensemble_parallel_coordinates.png')
    # plt.close()

    # log_loss_scores = log_loss_scores_df.sort_values(by='log_loss', ascending=True)
    # log_loss_scores.to_csv('outputs/predictive_modeling/classification/ensemble/ensemble_weighted_log_loss_scores.csv')
    # best_weights = log_loss_scores.iloc[0, :-1].values

    # ensemble_weighted = VotingClassifier(estimators=[('knn', knn), ('rf', rf), ('svm', svm), ('nb', nb), ('mlp', mlp)], voting='soft', n_jobs=-1, weights=best_weights)
    # ensemble_weighted.fit(X_train, y_train)
    # y_pred_weighted = ensemble_weighted.predict(X_test) + 1

    # joblib.dump(ensemble_weighted, 'outputs/predictive_modeling/classification/ensemble/ensemble_weighted_model.pkl')

    # # Metrics
    # accuracy_weighted = accuracy_score(y_test + 1, y_pred_weighted)
    # f1_weighted = f1_score(y_test + 1, y_pred_weighted, average='weighted')
    # precision_weighted = precision_score(y_test + 1, y_pred_weighted, average='weighted')
    # recall_weighted = recall_score(y_test + 1, y_pred_weighted, average='weighted')
    # log_loss_weighted = log_loss(y_test, ensemble_weighted.predict_proba(X_test))
    # metrics_weighted = {
    #     'accuracy': accuracy_weighted,
    #     'f1': f1_weighted,
    #     'precision': precision_weighted,
    #     'recall': recall_weighted,
    #     'logloss': log_loss_weighted
    # }
    # with open('outputs/predictive_modeling/classification/ensemble/ensemble_weighted_metrics.json', 'w') as f:
    #     json.dump(metrics_weighted, f)
    # print(metrics_weighted)





    # Create ensemble: stacking (using LightGBM)

    # Create data frame with training predictions of all models and real labels
    X_train_stacking = pd.DataFrame({
        'knn': knn.predict(X_train) + 1,
        'rf': rf.predict(X_train) + 1,
        'svm': svm.predict(X_train) + 1,
        'nb': nb.predict(X_train) + 1,
        'mlp': mlp.predict(X_train) + 1
    })

    X_val_stacking = pd.DataFrame({
        'knn': knn.predict(X_val) + 1,
        'rf': rf.predict(X_val) + 1,
        'svm': svm.predict(X_val) + 1,
        'nb': nb.predict(X_val) + 1,
        'mlp': mlp.predict(X_val) + 1
    })

    X_test_stacking = pd.DataFrame({
        'knn': knn.predict(X_test) + 1,
        'rf': rf.predict(X_test) + 1,
        'svm': svm.predict(X_test) + 1,
        'nb': nb.predict(X_test) + 1,
        'mlp': mlp.predict(X_test) + 1
    })

    # Preprocess stacking data
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Add knn, rf, svm, nb, mlp predictions as features to X_train from X_train_stacking
    for model in ['knn', 'rf', 'svm', 'nb', 'mlp']:
        X_train[model] = X_train_stacking[model]
        X_val[model] = X_val_stacking[model]
        X_test[model] = X_test_stacking[model]

    # Define the range of hyperparameters
    param_dist = {
        'n_estimators': randint(100, 2000),
        'learning_rate': uniform(0.01, 0.2),  
        'max_depth': randint(2, 50),  
        'num_leaves': randint(4, 200),  
        'reg_lambda': uniform(0.0001, 10.0),  
        'min_child_weight': randint(1, 15),
        'feature_fraction': uniform(0.01, 0.99),
        'bagging_fraction': uniform(0.4, 0.6),
        'bagging_freq': [1],  
    }

    # Hyperparameter tuning
    random_search = RandomizedSearchCV(estimator=lgb.LGBMClassifier(metric='multi_logloss', early_stopping_rounds=5, objective='multiclass'), 
                                param_distributions=param_dist, 
                                n_iter=1000, 
                                cv=10, 
                                verbose=2, 
                                scoring='neg_log_loss',
                                random_state=42, 
                                n_jobs=-1)
    random_search.fit(X_train, y_train,
                        eval_set=[(X_val, y_val)]
                    )

    # Save results of RandomizedSearchCV
    results = pd.DataFrame(random_search.cv_results_)
    interested_columns = ['param_' + param for param in param_dist.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results = results[interested_columns]
    results = results.sort_values(by='rank_test_score')
    results['mean_test_score'] = results['mean_test_score']
    results.to_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_cv_results.csv')

    # Parallel coordinate plot
    scaler = MinMaxScaler()
    results = results.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_num_leaves': 'num_leaves', 'param_reg_lambda': 'reg_lambda', 'param_min_child_weight': 'min_child_weight', 'param_feature_fraction': 'feature_fraction', 'param_bagging_fraction': 'bagging_fraction', 'param_bagging_freq': 'bagging_freq'})
    for param in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_lambda', 'min_child_weight', 'feature_fraction', 'bagging_fraction', 'bagging_freq']:
        results[param] = scaler.fit_transform(results[param].values.reshape(-1, 1))
    results = results.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results, 'mean_test_score', colormap='viridis', alpha = 0.25)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model, train_preds, test_preds, y_train, y_test = best_model_and_predictions(random_search, X_train, X_test, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_model.pkl', 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_hyperparameters.json', 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_train_preds.csv', 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_test_preds.csv')

    # Confusion matrix and metrics
    confusion_matrix_and_metrics(X_test, y_test, test_preds, best_model, 
                                cm_path = 'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_confusion_matrix.png',
                                scores_path = 'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_scores.json')
                                
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
        plt.savefig('outputs/predictive_modeling/classification/ensemble/roc_curve_class_{}.png'.format(cls))
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
        plt.savefig('outputs/predictive_modeling/classification/ensemble/precision_recall_curve_class_{}.png'.format(cls))
        plt.close()

    # Create feature importance plot for top 15
    feature_importance = best_model.feature_importances_
    features = X_train.columns
    feature_importance_scores = dict(zip(features, feature_importance))
    feature_importance_df = pd.DataFrame(feature_importance_scores.items(), columns=['Feature', 'Importance'])
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_feature_importance.csv', index=False)

    feature_importance_scores = dict(sorted(feature_importance_scores.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores.keys(), feature_importance_scores.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_feature_importance.png')
    plt.close()







    # Create ensemble: stacking (using LightGBM) - base models only without any ryanair data

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, datetime_train, datetime_val, datetime_test, data = create_pipeline('data/ryanair_reviews_with_extra_features.csv')

    # Define the range of hyperparameters
    param_dist2 = {
        'n_estimators': randint(100, 2000),
        'learning_rate': uniform(0.01, 0.2),  
        'max_depth': randint(2, 50),  
        'num_leaves': randint(4, 200),  
        'reg_lambda': uniform(0.0001, 10.0),  
        'min_child_weight': randint(1, 15),
        'feature_fraction': uniform(0.01, 0.99),
        'bagging_fraction': uniform(0.4, 0.6),
        'bagging_freq': [1],  
    }

    # Hyperparameter tuning
    random_search2 = RandomizedSearchCV(estimator=lgb.LGBMClassifier(metric='multi_logloss', early_stopping_rounds=1, objective='multiclass'),
                                param_distributions=param_dist2, 
                                n_iter=1000, 
                                cv=10, 
                                verbose=2, 
                                scoring='neg_log_loss',
                                random_state=42, 
                                n_jobs=-1)
    random_search2.fit(X_train_stacking, y_train,
                        eval_set=[(X_val_stacking, y_val)],
                    )

    # Save results of RandomizedSearchCV
    results2 = pd.DataFrame(random_search2.cv_results_)
    interested_columns2 = ['param_' + param for param in param_dist2.keys()] + ['mean_test_score', 'std_test_score', 'rank_test_score']
    results2 = results2[interested_columns2]
    results2 = results2.sort_values(by='rank_test_score')
    results2['mean_test_score'] = results2['mean_test_score']
    results2.to_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_cv_results.csv')

    # Parallel coordinate plot without max_features and bootstrap
    scaler2 = MinMaxScaler()
    results2 = results2.rename(columns={'param_n_estimators': 'n_estimators', 'param_learning_rate': 'learning_rate', 'param_max_depth': 'max_depth', 'param_num_leaves': 'num_leaves', 'param_reg_lambda': 'reg_lambda', 'param_min_child_weight': 'min_child_weight', 'param_feature_fraction': 'feature_fraction', 'param_bagging_fraction': 'bagging_fraction', 'param_bagging_freq': 'bagging_freq'})
    for param in ['n_estimators', 'learning_rate', 'max_depth', 'num_leaves', 'reg_lambda', 'min_child_weight', 'feature_fraction', 'bagging_fraction', 'bagging_freq']:
        results2[param] = scaler2.fit_transform(results2[param].values.reshape(-1, 1))
    results2 = results2.drop(columns=['std_test_score', 'rank_test_score'])
    plt.figure(figsize=(14, 7))
    parallel_coordinates(results2, 'mean_test_score', colormap='viridis', alpha=0.25)
    plt.legend().remove()
    plt.tight_layout()
    plt.savefig('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_parallel_coordinates.png')
    plt.close()
    # purple = best, yellow = worst

    # Best model, hyperparameters and predictions
    best_model2, train_preds2, test_preds2, y_train, y_test = best_model_and_predictions(random_search2, X_train_stacking, X_test_stacking, y_train, y_test, datetime_train, datetime_test, 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_model.pkl', 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_hyperparameters.json', 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_train_preds.csv', 
                            'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_test_preds.csv')

    # Confusion matrix and metrics
    confusion_matrix_and_metrics(X_test_stacking, y_test, test_preds2, best_model2, 
                                cm_path = 'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_confusion_matrix.png',
                                scores_path = 'outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_scores.json')
                                
    # Create ROC & PR curves for classes 1, 5, and 10
    probabilities2 = best_model2.predict_proba(X_test_stacking)
    classes2 = [1, 5, 10]

    for cls in classes2:
        # Create ROC curve
        fpr2, tpr2, _ = roc_curve((y_test == cls).astype(int), probabilities2[:, cls - 1])
        roc_auc2 = auc(fpr2, tpr2)

        plt.figure()
        plt.plot(fpr2, tpr2, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Class {})'.format(cls))
        plt.legend(loc="lower right")
        plt.savefig('outputs/predictive_modeling/classification/ensemble/roc_curve_class_{}_only_base.png'.format(cls))
        plt.close()

        # Create Precision-Recall curve
        precision2, recall2, _ = precision_recall_curve((y_test == cls).astype(int), probabilities2[:, cls - 1])

        plt.figure()
        plt.step(recall2, precision2, color='b', alpha=0.2, where='post')
        plt.fill_between(recall2, precision2, step='post', alpha=0.2, color='b')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Precision-Recall Curve (Class {})'.format(cls))
        plt.savefig('outputs/predictive_modeling/classification/ensemble/precision_recall_curve_class_{}_only_base.png'.format(cls))
        plt.close()

    # Create feature importance plot for top 15
    feature_importance2 = best_model2.feature_importances_
    features2 = X_train_stacking.columns
    feature_importance_scores2 = dict(zip(features2, feature_importance2))
    feature_importance_df2 = pd.DataFrame(feature_importance_scores2.items(), columns=['Feature', 'Importance'])
    feature_importance_df2 = feature_importance_df2.sort_values(by='Importance', ascending=False)
    feature_importance_df2.to_csv('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_feature_importance.csv', index=False)

    feature_importance_scores2 = dict(sorted(feature_importance_scores2.items(), key=lambda x: x[1], reverse=True)[:15])
    plt.figure(figsize=(14, 7))
    plt.bar(feature_importance_scores2.keys(), feature_importance_scores2.values())
    plt.ylabel('Feature Importance')
    plt.xlabel('Feature')
    plt.xticks(rotation=90)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('outputs/predictive_modeling/classification/ensemble/lgbm_ensemble_only_base_feature_importance.png')
    plt.close()



if __name__ == '__main__':
    main()