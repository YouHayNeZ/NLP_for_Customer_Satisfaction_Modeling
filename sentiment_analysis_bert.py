from transformers import pipeline
import pandas as pd


def preprocess_text(text):
    text = text.lower()
    return text

df = pd.read_csv('data/ryanair_reviews.csv')

comment_title = df['Comment title'].apply(preprocess_text).tolist()
comment = df['Comment'].apply(preprocess_text).tolist()
sentiment_analyzer = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
sentiment_comment_title = sentiment_analyzer(comment_title, truncation=True, max_length=512)
sentiment_comment = sentiment_analyzer(comment, truncation=True, max_length=512)

df['sentiment_comment_title'] = [result['label'] for result in sentiment_comment_title]
df['sentiment_score_comment_title'] = [result['score'] for result in sentiment_comment_title]

df['sentiment_comment'] = [result['label'] for result in sentiment_comment]
df['sentiment_score_comment'] = [result['score'] for result in sentiment_comment]

df.to_csv('data/ryanair_reviews_with_bert_sentiment.csv', index=False)

print(df.head())
