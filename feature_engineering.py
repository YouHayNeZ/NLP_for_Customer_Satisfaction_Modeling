# Feature Engineering

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load data
data = pd.read_csv('data/ryanair_reviews.csv')

# Regular feature engineering
def regular_feature_engineering(df):
    # Check if 'title' is "Ryanair customer review" (True/False) -> usually positive association
    df['ryanair_review'] = df['Comment title'].apply(lambda x: x == 'Ryanair customer review')

    # Count number of "!" in the 'comment' column (int) -> catch aggression (negative association)
    df['exclamation_marks'] = df['Comment'].apply(lambda x: x.count('!'))

    # Count number of "?" in the 'comment' column (int) -> catch sarcasm, rhetorical questions (negative association)
    df['question_marks'] = df['Comment'].apply(lambda x: x.count('?'))

    # Find the length of the 'comment' column (int) -> longer comments usually complain more (negative association)
    df['comment_length'] = df['Comment'].apply(lambda x: len(x))

    # Check if "never" is in the 'comment' column (True/False) -> usually negative association
    df['never'] = df['Comment'].apply(lambda x: 'never' in x)

    # Check if "again" is in the 'comment' column (True/False) -> usually negative association
    df['again'] = df['Comment'].apply(lambda x: 'again' in x)

    # Add more if you can think of anything
    #...
    #...
    #...

    # Normalize continuous variables
    scaler = StandardScaler()
    df[['exclamation_marks', 'question_marks', 'comment_length']] = scaler.fit_transform(df[['exclamation_marks', 'question_marks', 'comment_length']])
    
    return df

# Sentiment feature engineering
def sentiment_feature_engineering(df):
    #df['sentiment'] = ... # Sentiment analysis results
    return df

# Topic feature engineering
def topic_feature_engineering(df):
    #df['topic1'] = ... # Topic modeling results
    #df['topic2'] = ... # Topic modeling results
    #df['topic3'] = ... # Topic modeling results
    # ... add as many as needed
    return df

# Apply feature engineering
data = regular_feature_engineering(data)
data = sentiment_feature_engineering(data)
data = topic_feature_engineering(data)



