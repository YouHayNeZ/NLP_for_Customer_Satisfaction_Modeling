from transformers import pipeline
import pandas as pd


def preprocess_text(text):
    text = text.lower()
    return text


df = pd.read_csv('data/ryanair_reviews.csv')

comment = df['Comment'].apply(preprocess_text).tolist()

sentiment_analyzer_distilbert = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
sentiment_comment_distilbert = sentiment_analyzer_distilbert(comment, truncation=True, max_length=512)
df['sentiment_comment_distilbert'] = [result['label'] for result in sentiment_comment_distilbert]

sentiment_analyzer_siebert = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
sentiment_comment_siebert = sentiment_analyzer_distilbert(comment, truncation=True, max_length=512)
df['sentiment_comment_siebert'] = [result['label'] for result in sentiment_comment_siebert]

df.to_csv('data/ryanair_reviews_with_bert_sentiment.csv', index=False)