# Feature Engineering

import pandas as pd
import matplotlib.pyplot as plt

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
    openai_sentiment = sentiments_data['openai_sentiment']
    df['Sentiment'] = openai_sentiment
    df['Sentiment'] = df['Sentiment'].replace(-1, 'negative')
    df['Sentiment'] = df['Sentiment'].replace(0, 'neutral')
    df['Sentiment'] = df['Sentiment'].replace(1, 'positive')
    return df


# Topic feature engineering
def topic_feature_engineering(df):
    topics = pd.read_csv('outputs/nlp/topic_modeling/openai_topic_modeling.csv')
    """
    All 8 topics:
    Issues with luggage fees, Problems with boarding process, Good punctuality, Poor customer service, Comfort and seat space issues, Miscellaneous fees and charges, Delays and cancellations, Cleanliness and maintenance concerns
    """
    # Probability of the topics (dummies)
    df['topic_luggage'] = topics['topics'].apply(lambda x: 'Issues with luggage fees' in x)
    df['topic_boarding'] = topics['topics'].apply(lambda x: 'Problems with boarding process' in x)
    df['topic_punctual'] = topics['topics'].apply(lambda x: 'Good punctuality' in x)
    df['topic_service'] = topics['topics'].apply(lambda x: 'Poor customer service' in x)
    df['topic_comfort'] = topics['topics'].apply(lambda x: 'Comfort and seat space issues' in x)
    df['topic_other_fees'] = topics['topics'].apply(lambda x: 'Miscellaneous fees and charges' in x)
    df['topic_delay'] = topics['topics'].apply(lambda x: 'Delays and cancellations' in x)
    df['topic_clean'] = topics['topics'].apply(lambda x: 'Cleanliness and maintenance concerns' in x)
    return df


# Apply feature engineering
data = regular_feature_engineering(data)
data = sentiment_feature_engineering(data)
data = topic_feature_engineering(data)

# Save data
data.to_csv('data/ryanair_reviews_with_extra_features.csv', index=False)
