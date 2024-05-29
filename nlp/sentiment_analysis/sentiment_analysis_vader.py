import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# For preprocessing and sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def vader_add_sentiment_scores(df, original_col, cleaned_col):
    sia = SentimentIntensityAnalyzer()
    sia_cleaned = SentimentIntensityAnalyzer()

    # Add sentiment scores to the data frame for original comments
    df["score"] = df[original_col].apply(lambda x: sia.polarity_scores(x))
    df['negative'] = df['score'].apply(lambda x: x['neg'])
    df['neutral'] = df['score'].apply(lambda x: x['neu'])
    df['positive'] = df['score'].apply(lambda x: x['pos'])
    df['compound'] = df['score'].apply(lambda x: x['compound'])

    # Add sentiment scores to the data frame for cleaned comments
    df["score_cleaned"] = df[cleaned_col].apply(lambda x: sia_cleaned.polarity_scores(x))
    df['negative_cleaned'] = df['score_cleaned'].apply(lambda x: x['neg'])
    df['neutral_cleaned'] = df['score_cleaned'].apply(lambda x: x['neu'])
    df['positive_cleaned'] = df['score_cleaned'].apply(lambda x: x['pos'])
    df['compound_cleaned'] = df['score_cleaned'].apply(lambda x: x['compound'])

    # Drop the intermediate score columns
    df.drop(columns=["score", "score_cleaned"], inplace=True)

    return df

# Define the function for the emotion (positive, negative, neutral)
def vader_final_sentiment(compound):
    if compound > 0.05:
        return "positive"
    elif compound < -0.05:
        return "negative"
    elif compound >= -0.05 and compound <= 0.05:
        return "neutral"
    

def vader_plot_sentiment_distributions(comments, compound_col='compound', compound_cleaned_col='compound_cleaned'):
    """
    Plots the distribution of sentiment scores before and after cleaning.

    Parameters:
    comments (pd.DataFrame): DataFrame containing the sentiment scores.
    compound_col (str): Column name for the original compound scores.
    compound_cleaned_col (str): Column name for the cleaned compound scores.
    """
    
    # Create a figure and axes
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))

    # Plot the first plot
    axs[0].hist(comments[compound_col], bins=20, color='blue', alpha=0.7)
    axs[0].set_title("Distribution of the Sentiment Scores without Preprocessing")
    axs[0].set_xlabel('Compound Score')
    axs[0].set_ylabel('Frequency')

    # Plot the second plot
    axs[1].hist(comments[compound_cleaned_col], bins=20, color='green', alpha=0.7)
    axs[1].set_title("Distribution of the Sentiment Scores After Preprocessing")
    axs[1].set_xlabel('Compound Score')
    axs[1].set_ylabel('Frequency')

    # Show the combined plot
    plt.tight_layout()
    plt.show()


    
def vader_plot_sentiment_proportions(df, sentiment_col, sentiment_cleaned_col):
    # Calculate sentiment proportions
    sentiment_proportions = df[sentiment_col].value_counts() / len(df)
    sentiment_proportions_cleaned = df[sentiment_cleaned_col].value_counts() / len(df)
    
    # Generate pie charts
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the first pie chart
    axs[0].pie(
        sentiment_proportions,
        labels=sentiment_proportions.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0, 0],
        autopct='%1.1f%%'
    )
    axs[0].set_title("Sentiment Distribution Using Vader Sentiment Analysis")

    # Plot the second pie chart
    axs[1].pie(
        sentiment_proportions_cleaned,
        labels=sentiment_proportions_cleaned.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0, 0],
        autopct='%1.1f%%'
    )
    axs[1].set_title("Sentiment Distribution After Preprocessing")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the combined plot
    plt.show()

def vader_plot_rating_vs_compound(data, comments):
    """
    Analyzes and plots the relationship between overall ratings and VADER compound scores.

    Parameters:
        data (pd.DataFrame): The DataFrame containing ratings data.
        comments (pd.DataFrame): The DataFrame containing VADER compound scores.

    Returns:
        None
    """

    merged = pd.concat([comments, data["Overall Rating"]], axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(data=merged, x="Overall Rating", y="compound", ax=axs[0])
    axs[0].set_xlabel('Rating')
    axs[0].set_ylabel('Compound Score')
    axs[0].set_title('Rating vs. Compound Score')

    sns.boxplot(data=merged, x="Overall Rating", y="compound_cleaned", ax=axs[1])
    axs[1].set_xlabel('Rating')
    axs[1].set_ylabel('Compound Score After Preprocessing')
    axs[1].set_title('Rating vs. Compound Score After Preprocessing')

    correlation_cleaned = round(merged["Overall Rating"].corr(merged['compound_cleaned'], method='spearman'), 2)
    correlation = round(merged["Overall Rating"].corr(merged['compound'], method='spearman'), 2)

    axs[0].text(0.5, -0.15, f'Spearman correlation: {correlation:.2f}', ha='center', va='center', transform=axs[0].transAxes, fontsize=10, color='blue')
    axs[1].text(0.5, -0.15, f'Spearman correlation: {correlation_cleaned:.2f}', ha='center', va='center', transform=axs[1].transAxes, fontsize=10, color='blue')

    plt.tight_layout()
    plt.show()