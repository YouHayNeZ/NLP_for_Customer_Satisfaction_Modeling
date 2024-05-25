import pandas as pd
# import nltk
import seaborn as sns
import matplotlib.pyplot as plt
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
import spacy.cli
from textblob import TextBlob
import re


def read_data() -> (pd.DataFrame, pd.DataFrame):
    data = pd.read_csv("data/ryanair_reviews.csv")
    # data['Comment title'] = data['Comment title'].astype(pd.StringDtype())
    # data['Comment'] = data['Comment'].astype(pd.StringDtype())
    # print(data.head(), data.dtypes)
    comments = data[["Comment title", "Comment"]].copy()
    return data, comments


# Function to preprocess text using spaCy
def spacy_process(text):
    # Remove punctuation: This regular expression finds all characters that are not word characters (\w)
    # or whitespace (\s) and replaces them with an empty string, effectively removing punctuation.
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers: This regular expression finds all digit characters (\d) and replaces them with an empty string,
    # effectively removing all numeric characters from the text
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase: This ensures that all characters in the text are lowercase, making the text
    # case-insensitive.
    text = text.lower()

    # Tokenize the text: The spaCy model processes the text and returns a Doc object, which contains a sequence of tokens.
    doc = nlp(text)

    # Lemmatize and remove stopwords:
    # - token.lemma_: The lemma (base form) of the token.
    # - token.is_alpha: Checks if the token consists of alphabetic characters only.
    # - not token.is_stop: Checks if the token is not a stopword.
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Join tokens into a single string with spaces between them: This creates a single string from the list of tokens
    return ' '.join(tokens)


def textblob_sentiment(text):
    blob = TextBlob(text)
    return round(blob.sentiment.polarity, 2), blob.sentiment.subjectivity


def final_sentiment(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    elif -0.05 <= score <= 0.05:
        return "neutral"


def final_subjectivity(score):
    if score <= 0.50:
        return "objective"
    elif score > 0.50:
        return "subjective"


def plot_score_distribution(comments):
    # Creating 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Polarity of original comments
    axes[0, 0].hist(comments['polarity'], bins=20, color='blue', edgecolor='black')
    axes[0, 0].set_title('Polarity of Original Comments')
    axes[0, 0].set_xlabel('Polarity')
    axes[0, 0].set_ylabel('Frequency')

    # Polarity of cleaned comments
    axes[0, 1].hist(comments['cleaned_polarity'], bins=20, color='green', edgecolor='black')
    axes[0, 1].set_title('Polarity of Cleaned Comments')
    axes[0, 1].set_xlabel('Polarity')
    axes[0, 1].set_ylabel('Frequency')

    # Subjectivity of original comments
    axes[1, 0].hist(comments['subjectivity'], bins=20, color='red', edgecolor='black')
    axes[1, 0].set_title('Subjectivity of Original Comments')
    axes[1, 0].set_xlabel('Subjectivity')
    axes[1, 0].set_ylabel('Frequency')

    # Subjectivity of cleaned comments
    axes[1, 1].hist(comments['cleaned_subjectivity'], bins=20, color='purple', edgecolor='black')
    axes[1, 1].set_title('Subjectivity of Cleaned Comments')
    axes[1, 1].set_xlabel('Subjectivity')
    axes[1, 1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def plot_pie_distribution(comments):
    polarity_proportions = comments["polarity_group"].value_counts() / len(comments)
    polarity_proportions_cleaned = comments["polarity_group_cleaned"].value_counts() / len(comments)

    subjectivity_proportions = comments["subjectivity_group"].value_counts() / len(comments)
    subjectivity_proportions_cleaned = comments["subjectivity_group_cleaned"].value_counts() / len(comments)

    # Generate a pie chart to visualize the distribution of sentiment_proportions
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # Plot the first pie chart
    axs[0, 0].pie(
        polarity_proportions,
        labels=polarity_proportions.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0, 0],
        autopct='%1.1f%%'
    )
    axs[0, 0].set_title("Polarity Distribution Using TextBlob Sentiment Analysis")

    # Plot the second pie chart
    axs[0, 1].pie(
        polarity_proportions_cleaned,
        labels=polarity_proportions_cleaned.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0, 0],
        autopct='%1.1f%%'
    )
    axs[0, 1].set_title("Polarity Distribution After Cleaning")

    axs[1, 0].pie(
        subjectivity_proportions,
        labels=subjectivity_proportions.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0],
        autopct='%1.1f%%'
    )
    axs[1, 0].set_title("Subjectivity Distribution Using TextBlob Sentiment Analysis")

    axs[1, 1].pie(
        subjectivity_proportions_cleaned,
        labels=subjectivity_proportions_cleaned.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0],
        autopct='%1.1f%%'
    )
    axs[1, 1].set_title("Subjectivity Distribution After Cleaning")

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Show the combined plot
    plt.show()


def analyze_ratings_vs_sentiment(data, comments):
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_colwidth')

    merged = pd.concat([comments, data["Overall Rating"]], axis=1)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    sns.boxplot(data=merged, x="Overall Rating", y="polarity", ax=axs[0])
    axs[0].set_xlabel('Rating')
    axs[0].set_ylabel('Polarity Score')
    axs[0].set_title('Rating vs. Polarity Score')

    sns.boxplot(data=merged, x="Overall Rating", y="cleaned_polarity", ax=axs[1])
    axs[1].set_xlabel('Rating')
    axs[1].set_ylabel('Cleaned Polarity Score')
    axs[1].set_title('Rating vs. Cleaned Polarity Score')

    plt.tight_layout()
    plt.show()

    correlation_cleaned = merged['Overall Rating'].corr(merged['cleaned_polarity'], method='spearman')
    correlation = merged['Overall Rating'].corr(merged['polarity'], method='spearman')

    print("Correlation between Overall Rating and cleaned_polarity:", correlation_cleaned)  # 0.57
    print("Correlation between Overall Rating and polarity:", correlation)  # 0.61


if __name__ == '__main__':
    # Change pandas settings to display all columns
    # pd.set_option('display.max_columns', None)
    # Adjust pandas settings to display the full content of each column
    # pd.set_option('display.max_colwidth', None)
    data, comments = read_data()
    # Helper function to apply func to a pandas Series
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    comments["cleaned_Comment"] = comments["Comment"].apply(spacy_process)
    # comments.loc[:, 'cleaned_Comment'] = comments['Comment'].apply(spacy_process)
    comments.to_csv('data/cleaned_comments.csv')

    comments = pd.read_csv('cleaned_comments.csv')
    comments['polarity'], comments['subjectivity'] = zip(*comments['Comment'].apply(textblob_sentiment))

    comments['cleaned_polarity'], comments['cleaned_subjectivity'] = zip(
        *comments['cleaned_Comment'].apply(textblob_sentiment))

    comments["polarity_group"] = comments["polarity"].apply(lambda compound: final_sentiment(compound))

    comments["polarity_group_cleaned"] = comments["cleaned_polarity"].apply(lambda compound: final_sentiment(compound))

    comments["subjectivity_group"] = comments["subjectivity"].apply(lambda compound: final_subjectivity(compound))

    comments["subjectivity_group_cleaned"] = comments["cleaned_subjectivity"].apply(
        lambda compound: final_subjectivity(compound))

    # comments.to_csv('data/textblob_comments.csv')

    plot_score_distribution(comments)
    plot_pie_distribution(comments)

    # comments.drop(columns=["score"], inplace=True)
    # comments.drop(columns=["score_cleaned"], inplace=True)
    # data_vader = add_vader_scores(data)
    # data_vader_textblob = add_textblob_scores(data_vader)

    # plot_vader_scores(data_vader)
    # plot_vader_scores(data_vader_textblob)
    # plot_textblob_scores(data_vader_textblob)

    # Print the comments with a VADER_Comment_compound score higher than 0.9
    # high_score_comments = data_vader[data_vader['VADER_Comment_compound'] > 0.9]['Comment']
    # print("Comments with VADER_Comment_compound score higher than 0.9:")
    # print(high_score_comments)

    # Print the comments with a VADER_Comment_compound score higher than 0.9
    # low_score_comments = data_vader[data_vader['VADER_Comment_compound'] < -0.9]['Comment']
    # print("Comments with VADER_Comment_compound score higher than 0.9:")
    # print(low_score_comments)

    # Filter the dataframe for rows where sentiment and sentiment_cleaned are different
    different_sentiments = comments[comments["polarity_group"] != comments["polarity_group_cleaned"]]

    """pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    # Access the "cleaned_Comment" and "Comment" columns for the filtered rows
    difference = different_sentiments[["polarity_group", "Comment", "polarity_group_cleaned", "cleaned_Comment"]]
    difference.to_csv("tx_sentiment_difference.csv")"""

    analyze_ratings_vs_sentiment(data, comments)


"""
spacy.cli.download("en_core_web_sm")
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

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

    def textblob_polarity(text):
        return TextBlob(text).sentiment.polarity
"""
