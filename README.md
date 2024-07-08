# Topic Trends and Travel Tones: Decoding Ryanair Reviews

## Table of Contents
- [Project Overview](#project-overview)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Description](#model-description)
- [Results](#results)
- [Discussion and Business Implications](#discussion-and-business-implications)
- [Limitations](#limitations)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview
This project aims to analyze customer reviews of Ryanair using advanced Natural Language Processing (NLP) techniques and predictive modeling. The analysis includes sentiment analysis, topic modeling, and predictive modeling to enhance understanding of customer satisfaction and provide actionable recommendations.

## Motivation
Customer feedback is crucial for understanding and improving service quality. Analyzing Ryanair's customer reviews can help identify common issues, predict customer satisfaction, and provide actionable recommendations for service improvement. This project integrates sentiment analysis and topic modeling, leveraging these results for predictive modeling.

## Dataset
### Source
The dataset consists of Ryanair customer reviews obtained from Kaggle.

### Description
- **Raw Data:** `data/ryanair_reviews.csv`
- **Data with Extra Features:** `data/ryanair_review_with_extra_features.csv`

### Project Structure
```
.
├── data                                -> Contains raw and processed data
│   ├── ryanair_review_with_extra_features.csv   -> Preprocessed and cleaned data with included features from NLP
│   └── ryanair_reviews.csv                      -> Raw data
├── nlp                                 -> NLP analysis scripts
│   ├── sentiment_analysis              
│   │   ├── annotation_of_test_data_benchmark_analysis.py   -> manual annotation of comments for sentiment benchmark analysis
│   │   ├── benchmarking_helper_functions.py
│   │   ├── sentiment_analysis_bert.py
│   │   ├── sentiment_analysis_openai.py
│   │   ├── sentiment_analysis_textblob.py
│   │   ├── sentiment_analysis_vader.py
│   │   └── sentiment_benchmark_analysis.ipynb  -> conducted the different sentiment analyses and the benchmark analysis there
│   ├── topic_modeling
│   │   ├── topic_modeling_openai.py    
│   │   ├── topic_modeling.ipynb
│   │   ├── topic_modelling_LDA.py
│   │   ├── topics_first_run
│   │   └── topics_second_run
│   └── comments_preprocessing.py   -> pipeline for comments preprocessing
├── predictive_modeling                 -> Predictive modeling scripts
│   ├── classification
│   │   ├── base_learners
│   │   │   ├── training_knn_classif.py
│   │   │   ├── training_mlp_classif.py
│   │   │   ├── training_nb_classif.py
│   │   │   ├── training_rf_classif.py
│   │   │   └── training_svm_classif.py
│   │   └── ensemble_classif.py
│   ├── regression
│   │   ├── base_learners
│   │   │   ├── training_bayesian_ridge_reg.py
│   │   │   ├── training_knn_reg.py
│   │   │   ├── training_mlp_reg.py
│   │   │   ├── training_rf_reg.py
│   │   │   └── training_svm_reg.py
│   │   └── ensemble_reg.py
│   ├── benchmarking.py   
│   └── training_helper_func.py
├── outputs                             -> Outputs generated from analysis
│   ├── nlp
│   │   ├── sentiment_analysis
│   │   │   ├── bert_sentiment_analysis.csv     -> easier access to save time
│   │   │   ├── cleaned_comments.csv            -> easier access to save time
│   │   │   ├── labeled_data_for_benchmark_analysis.csv
│   │   │   └── openai_sentiment_analysis.csv
│   │   ├── topic_modeling
│   │   │   ├── coherence_vs_perplexity.png
│   │   │   ├── comments_with_lda_topics.csv
│   │   │   ├── lda_topics_visualization.html
│   │   │   ├── lda_topics.png
│   │   │   └── openai_topic_modeling.csv
│   ├── predictive_modeling
│   │   ├── classification
│   │   │   ├── base_learners
│   │   │   │   ├── knn
│   │   │   │   │   ├── knn_confusion_matrix.png
│   │   │   │   │   ├── knn_cv_results.csv
│   │   │   │   │   ├── knn_hyperparameters.json
│   │   │   │   │   ├── knn_model.pkl
│   │   │   │   │   ├── knn_parallel_coordinates.png
│   │   │   │   │   ├── knn_scores.json
│   │   │   │   │   ├── knn_test_preds.csv
│   │   │   │   │   ├── knn_train_preds.csv
│   │   │   │   │   ├── precision_recall_curve_class_1.png
│   │   │   │   │   ├── precision_recall_curve_class_5.png
│   │   │   │   │   └── precision_recall_curve_class_10.png
│   │   │   │   │   ├── roc_curve_class_1.png
│   │   │   │   │   ├── roc_curve_class_5.png
│   │   │   │   │   └── roc_curve_class_10.png
│   │   │   │   ├── mlp -> Same structure as knn
│   │   │   │   ├── nb -> Same structure as knn
│   │   │   │   ├── rf -> Same structure as knn
│   │   │   │   ├── svm -> Same structure as knn
│   │   │   ├── ensemble -> Same as base learners plus additional graphs and illustration
│   │   │   ├── feature_selection
│   │   │   │   ├── feature_importance_image.png
│   │   │   │   └── feature_importance_scores.csv
│   │   ├── regression
│   │   │   ├── base_learners
│   │   │   │   ├── bayesian_ridge
│   │   │   │   │   ├── bayesian_ridge_cv_results.csv
│   │   │   │   │   ├── bayesian_ridge_hyperparameters.json
│   │   │   │   │   ├── bayesian_ridge_model.pkl
│   │   │   │   │   ├── bayesian_ridge_parallel_coordinates.png
│   │   │   │   │   ├── bayesian_ridge_scores.json
│   │   │   │   │   ├── bayesian_ridge_test_preds.csv
│   │   │   │   │   ├── bayesian_ridge_train_predictions.png
│   │   │   │   │   └── bayesian_ridge_train_preds.csv
│   │   │   │   ├── knn -> Same structure as bayesian_ridge
│   │   │   │   ├── mlp -> Same structure as bayesian_ridge
│   │   │   │   ├── rf -> Same structure as bayesian_ridge
│   │   │   │   ├── svm -> Same structure as bayesian_ridge
│   │   │   ├── ensemble -> Same as base learners plus additional
│   │   │   ├── feature_selection
│   │   │   │   ├── feature_importance_image.png
│   │   │   │   └── feature_importance_scores.csv
│   │   ├── model_comparison_classif_vs_reg.csv
│   │   ├── model_comparison_test_preds_with_vs_without_FE.csv
├── adressed_topics.txt
├── exploration.ipynb
├── feature_engineering.py
├── preprocessing.py
├── README.md
├── requirements.txt
```

## Installation
To install and set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/YouHayNeZ/Python4Business.git
    ```

2. (Optional) Create a virtual environment where the packages should be installed and activate it:
    ```bash
    conda create -n python4business python=3.10
    conda activate python4business
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
### Preprocessing for Sentiment Analysis
To preprocess the comments, run:
```bash
python nlp/comments_preprocessing.py
```

### Sentiment Analysis
To perform sentiment analysis using different models, navigate to the following path and run the jupyter notebook:
```
nlp/sentiment_analysis/sentiment_benchmark_analysis.ipynb
```

### Topic Modeling
To perform topic modeling using LDA and OpenAI models, run:

# LDA
```bash
python nlp/topic_modeling/topic_modelling_LDA.py
```

# OpenAI GPT-3.5-turbo
Note that you need a key to re-run the script.
```bash
python nlp/topic_modeling/topic_modeling_openai.py
```

# Dashboard
To run the dashboard that visualizes the results of Sentiment Analysis and Topic Modeling, navigate to the following path and run the jupyter notebook:

```
nlp/topic_modeling/dashboard.ipynb
```

### Predicive Modeling
To train and evaluate predictive models, run the respective training .py files
The illustrations and graphs are then automatically loaded in the respective folder.

Example for Ensemble Classifier:
```bash
python predictive_modeling/classification/ensemble_classif.py
```
The illustrations and predictions will be saved in the folder `outputs/predictive_modeling/classification/ensemble`.

## Model Description
### Sentiment Analysis
- **Models Used:** VADER, TextBlob, DistilBERT, RoBERTa, OpenAI GPT-3.5-turbo
- **Preprocessing:** Lowercasing, removing stopwords, tokenization, etc.

### Topic Modeling
- **Models Used:** Latent Dirichlet Allocation (LDA), OpenAI GPT-3.5-turbo
- **Evaluation:** Coherence score, qualitative analysis

### Predictive Modeling
- **Tasks:** Classification, Regression
- **Base Learners for Classification:** MLP, KNN, Naive Bayes, Random Forest, SVM
- **Base Learners for Regression:** Bayesian Ridge, KNN, MLP, Random Forest SVM
- **Ensemble Methods:** Unweighted and weighted majority voting, stacking with LightGBM (both for classification and regression)

## Results
### Sentiment Analysis
- **Best performing model:** OpenAI GPT-3.5-turbo
- **Metrics:** Accuracy, Precision, Recall, F1 Score

### Topic Modeling
- **Best performing model (based on qualitative analysis):** OpenAI GPT-3.5-turbo
- **Topics identified:** Passenger experience, charges, luggage issues, delays, customer service, etc.

### Predictive Modeling
- **Classification:**
  - **Best performing models:** LGBM Ensemble, Majority Voting (Weighted), Random Forest (Base learner)
  - **Key Findings:** The effectiveness of ensemble methods underscores the value of combining multiple models to leverage their strengths and mitigate individual weaknesses. This approach is particularly beneficial in complex prediction tasks involving customer reviews
  - **Metrics:** Accuracy, Precision, Recall, F1 Score, Log Loss

- **Regression:**
  - **Best performing model:** Random Forest
  - **Key Findings:** Random Forest excells in regression tasks, offering the best fit for predicting numerical outcomes related to customer satisfaction. It showed the lowest error rates and highest predictive power among the regression models tested.
  - **Metrics:** MSE, R2, MAE


## Discussion and Business Implications
The analysis provides actionable insights into customer satisfaction. By understanding common topics and sentiments, Ryanair can improve specific aspects of their service. The predictive modeling helps identify key factors influencing overall customer satisfaction.

## Limitations
- The dataset is biased towards extreme positive or negative experiences.
- Use of OpenAI models may incur costs and their non-deterministic nature could lead to variability in results.
- The analysis is limited to English comments.


## License
This project is licensed under the MIT License.

## Contact
For questions or feedback, please contact:
- **Patrycja Slotosch:** [patrycja.slotosch@tum.de](mailto:patrycja.slotosch@tum.de)
- **Bilge Ece Gundogdu:** [b.guendogdu@tum.de](mailto:b.guendogdu@tum.de)
- **Johannes Laurin Vorbach:** [Johannes.Vorbach@tum.de](mailto:Johannes.Vorbach@tum.de)
- **Niklas Kothe:** [niklas.kothe@tum.de](mailto:niklas.kothe@tum.de)
