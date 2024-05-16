import pandas as pd
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob


def read_data() -> pd.DataFrame:
    data = pd.read_csv("data/ryanair_reviews.csv")
    data['Comment title'] = data['Comment title'].astype(pd.StringDtype())
    data['Comment'] = data['Comment'].astype(pd.StringDtype())
    # print(data.head(), data.dtypes)
    return data


def add_vader_scores(data: pd.DataFrame) -> pd.DataFrame:
    # nltk.download('vader_lexicon')
    # Initialize the VADER sentiment analyzer
    analyzer = SentimentIntensityAnalyzer()
    # Apply the function to 'Comment' and 'Comment title' columns and store the results in new columns
    data['VADER_Comment'] = data['Comment'].apply(lambda x: analyzer.polarity_scores(x))
    data['VADER_Title'] = data['Comment title'].apply(lambda x: analyzer.polarity_scores(x))

    # Expand the dictionary columns into separate columns for better readability
    vader_comment_df = data['VADER_Comment'].apply(pd.Series)
    vader_title_df = data['VADER_Title'].apply(pd.Series)

    # Join the new columns back to the original DataFrame
    data = pd.concat([data, vader_comment_df.add_prefix('VADER_Comment_'), vader_title_df.add_prefix('VADER_Title_')],
                   axis=1)
    return data


def plot_vader_scores(data: pd.DataFrame):
    # Plotting the distributions
    plt.figure(figsize=(14, 6))

    # Plot for VADER_Comment_compound
    plt.subplot(1, 2, 1)
    sns.histplot(data['VADER_Comment_compound'], bins=20, kde=True, color='blue')
    plt.title('Distribution of VADER Comment Compound Scores')
    plt.xlabel('VADER Comment Compound Score')
    plt.ylabel('Frequency')
    plt.xlim(-1, 1)

    # Plot for VADER_Title_compound
    plt.subplot(1, 2, 2)
    sns.histplot(data['VADER_Title_compound'], bins=20, kde=True, color='green')
    plt.title('Distribution of VADER Title Compound Scores')
    plt.xlabel('VADER Title Compound Score')
    plt.ylabel('Frequency')
    plt.xlim(-1, 1)

    plt.tight_layout()
    plt.show()


def add_textblob_scores(data: pd.DataFrame) -> pd.DataFrame:
    # Apply TextBlob sentiment analysis to 'Comment' and 'Comment title' columns
    data['TextBlob_Comment_Polarity'] = data['Comment'].apply(lambda x: TextBlob(x).sentiment.polarity)
    data['TextBlob_Title_Polarity'] = data['Comment title'].apply(lambda x: TextBlob(x).sentiment.polarity)
    return data


def plot_textblob_scores(data: pd.DataFrame):
    # Plotting the distributions
    plt.figure(figsize=(14, 6))

    # Plot for TextBlob_Comment_Polarity
    plt.subplot(1, 2, 1)
    sns.histplot(data['TextBlob_Comment_Polarity'], bins=20, kde=True, color='purple')
    plt.title('Distribution of TextBlob Comment Polarity Scores')
    plt.xlabel('TextBlob Comment Polarity Score')
    plt.ylabel('Frequency')
    plt.xlim(-1, 1)

    # Plot for TextBlob_Title_Polarity
    plt.subplot(1, 2, 2)
    sns.histplot(data['TextBlob_Title_Polarity'], bins=20, kde=True, color='orange')
    plt.title('Distribution of TextBlob Title Polarity Scores')
    plt.xlabel('TextBlob Title Polarity Score')
    plt.ylabel('Frequency')
    plt.xlim(-1, 1)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Change pandas settings to display all columns
    # pd.set_option('display.max_columns', None)
    # Adjust pandas settings to display the full content of each column
    pd.set_option('display.max_colwidth', None)
    data = read_data()
    data_vader = add_vader_scores(data)
    data_vader_textblob = add_textblob_scores(data_vader)

    # plot_vader_scores(data_vader)
    plot_vader_scores(data_vader_textblob)
    plot_textblob_scores(data_vader_textblob)

    # Print the comments with a VADER_Comment_compound score higher than 0.9
    # high_score_comments = data_vader[data_vader['VADER_Comment_compound'] > 0.9]['Comment']
    # print("Comments with VADER_Comment_compound score higher than 0.9:")
    # print(high_score_comments)

    # Print the comments with a VADER_Comment_compound score higher than 0.9
    # low_score_comments = data_vader[data_vader['VADER_Comment_compound'] < -0.9]['Comment']
    # print("Comments with VADER_Comment_compound score higher than 0.9:")
    # print(low_score_comments)
