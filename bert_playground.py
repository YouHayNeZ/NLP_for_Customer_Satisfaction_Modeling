import pandas as pd

pd.set_option('display.max_rows', None)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 200)  # Set the display width to avoid truncation
pd.set_option('display.max_colwidth', 500)  # Set column width to show long text

df = pd.read_csv('data/ryanair_reviews_with_bert_sentiment.csv')
mask = df['sentiment_comment_s'] != df['sentiment_comment_d']
different_labels_df = df[mask]
print(different_labels_df[['Comment', 'sentiment_comment_s', 'sentiment_comment_d']])
