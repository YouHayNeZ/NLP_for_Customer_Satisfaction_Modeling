# Feature Engineering

import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('data/ryanair_reviews.csv')
data = data.dropna(subset=['Overall Rating'])

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

    # Check if "online" is in the 'comment' column (True/False) -> usually positive association
    df['online'] = df['Comment'].apply(lambda x: 'online' in x)

    # Check if "delayed" is in the 'comment' column (True/False) -> usually negative association
    df['delayed'] = df['Comment'].apply(lambda x: 'delayed' in x)

    # Check if "cheap" is in the 'comment' column (True/False) -> usually positive association
    df['cheap'] = df['Comment'].apply(lambda x: 'cheap' in x)

    # Check if "legroom" is in the 'comment' column (True/False) -> usually negative association
    df['legroom'] = df['Comment'].apply(lambda x: 'legroom' in x)
    return df

# Sentiment feature engineering
def sentiment_feature_engineering(df):
    sentiments_data = pd.read_csv('outputs/nlp/sentiment_analysis/openai_sentiment_analysis.csv')
    sentiments_data = sentiments_data.dropna(subset=['Overall Rating'])
    openai_sentiment = sentiments_data['openai_sentiment']
    df['Sentiment'] = openai_sentiment
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

# Save data (TOPIC MODELING TO BE ADDED!)
data.to_csv('data/ryanair_reviews_some_features.csv', index=False)