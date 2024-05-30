import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from preprocessing import *
import pandas as pd

data = import_data('data/ryanair_reviews.csv')
data = drop_missing_target(data)

# Splitting the dataset into the Training set and Test set
X = data[['Comment title', 'Comment']]
y = data['Overall Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# combine X_test and y_test again
data_test = pd.concat([X_test, y_test], axis=1)

# Annotate test data
annotations_1_118 = [1, -1, -1, -1, -1, -1, 1, 1, -1, 1,
                     -1, 1, 1, 1, -1, -1, 1, -1, 1, 0,
                     -1, 0, -1, -1, 1, -1, -1, 0, -1, 0,
                     -1, 1, -1, -1, 1, -1, 1, 1, -1, -1,
                     -1, -1, -1, -1, 1, 1, -1, 1, -1, -1,
                     1, 1, -1, 1, 1, 1, -1, -1, -1, -1,
                     1, 1, -1, 0, -1, -1, -1, -1, -1, -1,
                     1, -1, 0, 1, 1, -1, 1, 1, 1, -1,
                     1, 1, -1, -1, -1, -1, 1, -1, -1, -1,
                     0, -1, -1, -1, 1, -1, -1, -1, 1, -1,
                     0, -1, -1, -1, -1, 1, 1, -1, -1, -1,
                     -1, 1, 1, -1, -1, 1, 1]

annotations_119_238 = [1, 1, -1, 1, -1, -1, -1, 1, -1, 1, 1, -1,
                       -1, -1, -1, -1, -1, 1, 1, -1, -1, 1,
                       -1, -1, -1, 0, -1, -1, 0, -1, 1, -1,
                       1, -1, -1, 0, 1, 1, 1, -1, 0, 0,
                       -1, -1, 1, 1, 1, -1, 0, -1, 1, 1,
                       -1, 1, -1, -1, -1, 0, -1, -1, -1, 1,
                       -1, 1, -1, 1, 1, -1, -1, 1, -1, 0,
                       -1, 1, 1, 1, 1, 1, -1, 1, 1, -1,
                       -1, 1, -1, 0, 0, -1, 1, 0, -1, -1,
                       -1, -1, 1, -1, -1, -1, -1, 1, -1, -1,
                       -1, -1, -1, 1, -1, -1, -1, 1, -1, -1,
                       1, -1, -1, 1, -1, -1, -1, -1]

annotations_239_355 = [-1, 1, 1, -1, 0, 1, 1, 1,-1, -1,
                      -1, 1, 1, -1, 1, 1, -1, -1, 1, 1,
                      1, -1, 0, -1, -1, 0, -1, -1, -1, -1,
                      -1, 0, 1, 1, -1, -1, -1, -1, 0, 1,
                      -1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                      -1, -1, -1, -1, 1, -1, -1, -1, 1, -1,
                      1, 1, -1, -1, 1, 1, -1, -1, -1, 1,
                      -1, 1, 1, -1, -1, -1, 1, -1, -1, -1,
                      0, 0, -1, -1, 1, 1, 1, -1, -1, 1,
                      1, -1, -1, -1, 0, 1, -1, -1, -1, -1,
                      0, 0, -1, 1, -1, -1, -1, -1, -1, -1,
                      -1, 1, -1, -1, -1, 1, 1] 

annotations_356_472 = [0, 1, -1, 1, 1, 1, -1, -1,
                       1, -1, -1, -1, -1, 1, -1, -1, 1, 1,
                       1, 1, 1, 1, -1, 1, 1, 1, 0, 1,
                       -1, 1, 1, -1, 1, 1, 1, -1, 1, -1,
                       -1, -1, -1, -1, 1, -1, -1, 0, 1, 1,
                       1, -1, 1, -1, 1, 1, 1, -1, 1, -1,
                       0, -1, -1, 0, -1, 1, 1, -1, 0, -1,
                       1, -1, -1, -1, -1, -1, 1, -1, -1, -1,
                       1, 0, 1, -1, 1, 1, -1, -1, -1, 1,
                       1, -1, 1, 1, -1, -1, -1, 1, -1, -1,
                       -1, -1, 1, 1, 1, 1, -1, -1, - 1, -1,
                       -1, -1, 1, -1, 1, 0, 0]

# Create pandas Series from the lists
annotations_1_118_series = pd.Series(annotations_1_118, name="annotations_1_118")
annotations_119_238_series = pd.Series(annotations_119_238, name="annotations_119_238")
annotations_239_355_series = pd.Series(annotations_239_355, name="annotations_239_355")
annotations_356_472_series = pd.Series(annotations_356_472, name="annotations_356_472")

# Combine the Series with test data into a DataFrame
annotations = pd.concat([annotations_1_118_series, annotations_119_238_series, annotations_239_355_series, annotations_356_472_series], axis=0)
annotations.reset_index(drop = True, inplace = True)

data_test.reset_index(drop=True, inplace = True)

# Combine them into a single DataFrame
annotations_df = pd.concat([data_test, annotations], axis=1)
annotations_df.columns = list(data_test.columns) + ['annotations']

# save to csv
annotations_df.to_csv("outputs/nlp/sentiment_analysis/labeled_data_for_benchmark_analysis.csv", index=False)