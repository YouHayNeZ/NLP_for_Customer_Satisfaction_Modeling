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
    lda_topic_modeling_data = pd.read_csv('outputs/nlp/topic_modeling/comments_with_lda_topics.csv')
    """
    Topic 1: passenger, plane, staff, board, sit, seat, cabin_crew, come, boarding, aircraft
    Topic 2: pay, seat, airline, book, check, charge, ticket, cheap, people, return
    Topic 3: bag, pay, check, luggage, hold, hand_luggage, staff, priority, queue, baggage
    Topic 4: time, delay, hour, plane, minute, wait, late, arrive, leave, people
    Topic 5: tell, check, airport, try, customer_service, hour, book, refund, ask, cancel
    Topic 6: time, seat, good, service, staff, price, airline, return, great, problem
    Topic 7: time, crew, good, return, cabin_crew, boarding, land, arrive, leg_room, journey
    """
    # Probability of the topics - numeric
    df['topic1'] = lda_topic_modeling_data['Topic_1_probability']
    df['topic2'] = lda_topic_modeling_data['Topic_2_probability']
    df['topic3'] = lda_topic_modeling_data['Topic_3_probability']
    df['topic4'] = lda_topic_modeling_data['Topic_4_probability']
    df['topic5'] = lda_topic_modeling_data['Topic_5_probability']
    df['topic6'] = lda_topic_modeling_data['Topic_6_probability']
    df['topic7'] = lda_topic_modeling_data['Topic_7_probability']
    # Name of the topic with the highest probability - string
    df['topic'] = lda_topic_modeling_data['Max_Probability_Topic']
    # Keywords of tha topic - string
    df['topic_keywords'] = lda_topic_modeling_data['Max_Topic_Words']
    return df


# Apply feature engineering
data = regular_feature_engineering(data)
data = sentiment_feature_engineering(data)
data = topic_feature_engineering(data)

# Save data
data.to_csv('data/ryanair_reviews_with_extra_features.csv', index=False)
