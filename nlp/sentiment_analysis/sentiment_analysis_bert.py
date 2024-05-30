from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt


def preprocess_text_for_bert(text):
    """
    Preprocesses the input text by converting it to lowercase.

    Parameters:
        text (str): The input text to preprocess.

    Returns:
        str: The preprocessed text.
    """
    return text.lower()


def analyze_sentiments():
    """
    Reads a CSV file, preprocesses the comments, performs sentiment analysis using two different models,
    and saves the results to a new CSV file.

    Parameters:
        input_csv (str): The path to the input CSV file containing the comments.
        output_csv (str): The path to the output CSV file to save the results.
    """
    # Read the CSV file
    df = pd.read_csv('data/ryanair_reviews.csv')

    # Preprocess the comments
    comments = df['Comment'].apply(preprocess_text_for_bert).tolist()

    # Sentiment analysis using distilbert model
    sentiment_analyzer_distilbert = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    sentiment_comment_distilbert = sentiment_analyzer_distilbert(comments, truncation=True, max_length=512)
    df['sentiment_comment_distilbert'] = [result['label'] for result in sentiment_comment_distilbert]

    # Sentiment analysis using siebert model
    sentiment_analyzer_siebert = pipeline('sentiment-analysis', model='siebert/sentiment-roberta-large-english')
    sentiment_comment_siebert = sentiment_analyzer_siebert(comments, truncation=True, max_length=512)
    df['sentiment_comment_siebert'] = [result['label'] for result in sentiment_comment_siebert]

    # Save the results to a new CSV file
    df.to_csv('outputs/nlp/sentiment_analysis/bert_sentiment_analysis.csv', index=False)

def bert_plot_sentiment_proportions(df, distilbert_col, siebert_col):
    """
    Plots the sentiment proportions from the given DataFrame columns using pie charts.

    Parameters:
        df (pd.DataFrame): The DataFrame containing sentiment data.
        distilbert_col (str): The column name in the DataFrame which contains DistilBERT sentiment classifications.
        siebert_col (str): The column name in the DataFrame which contains SieBERT sentiment classifications.

    Returns:
        None
    """
    sentiment_labels = {"POSITIVE": 'positive', "NEGATIVE": 'negative', "NEUTRAL": 'neutral'}

    # Calculate sentiment proportions
    sentiment_proportions_distilbert = df[distilbert_col].value_counts(normalize=True)
    sentiment_proportions_siebert = df[siebert_col].value_counts(normalize=True)
    
    # Map the sentiment values to their corresponding labels
    distilbert_labels = [sentiment_labels[val] for val in sentiment_proportions_distilbert.index]
    siebert_labels = [sentiment_labels[val] for val in sentiment_proportions_siebert.index]

    # Generate pie charts
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first pie chart
    axs[0].pie(
        sentiment_proportions_distilbert,
        labels=distilbert_labels,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1 if i == 0 else 0 for i in range(len(distilbert_labels))],  # Explode the first slice
        autopct='%1.1f%%'
    )
    axs[0].set_title("Sentiment Distribution with DistilBERT")

    # Plot the second pie chart
    axs[1].pie(
        sentiment_proportions_siebert,
        labels=siebert_labels,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1 if i == 0 else 0 for i in range(len(siebert_labels))],  # Explode the first slice
        autopct='%1.1f%%'
    )
    axs[1].set_title("Sentiment Distribution with SieBERT")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the combined plot
    plt.show()

if __name__ == '__main__':
    analyze_sentiments()