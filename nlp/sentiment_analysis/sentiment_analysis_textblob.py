import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import spacy.cli
from textblob import TextBlob
import re


def read_data() -> (pd.DataFrame, pd.DataFrame):
    data = pd.read_csv("data/ryanair_reviews.csv")
    # data['Comment title'] = data['Comment title'].astype(pd.StringDtype())
    # data['Comment'] = data['Comment'].astype(pd.StringDtype())
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


def textblob_final_sentiment(score):
    if score > 0.05:
        return "positive"
    elif score < -0.05:
        return "negative"
    elif -0.05 <= score <= 0.05:
        return "neutral"


def textblob_final_subjectivity(score):
    if score <= 0.50:
        return "objective"
    elif score > 0.50:
        return "subjective"


def textblob_plot_score_distribution(comments):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Polarity of original comments
    axes[0, 0].hist(comments['polarity'], bins=20, color='blue', edgecolor='black')
    axes[0, 0].set_title('Polarity of Original Comments')
    axes[0, 0].set_xlabel('Polarity')
    axes[0, 0].set_ylabel('Frequency')

    # Polarity of cleaned comments
    axes[0, 1].hist(comments['cleaned_polarity'], bins=20, color='green', edgecolor='black')
    axes[0, 1].set_title('Polarity of Preprocessed Comments')
    axes[0, 1].set_xlabel('Polarity')
    axes[0, 1].set_ylabel('Frequency')

    # Subjectivity of original comments
    axes[1, 0].hist(comments['subjectivity'], bins=20, color='red', edgecolor='black')
    axes[1, 0].set_title('Subjectivity of Original Comments')
    axes[1, 0].set_xlabel('Subjectivity')
    axes[1, 0].set_ylabel('Frequency')

    # Subjectivity of cleaned comments
    axes[1, 1].hist(comments['cleaned_subjectivity'], bins=20, color='purple', edgecolor='black')
    axes[1, 1].set_title('Subjectivity of Preprocessed Comments')
    axes[1, 1].set_xlabel('Subjectivity')
    axes[1, 1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def textblob_plot_pie_distribution(comments):
    polarity_proportions = comments["polarity_group"].value_counts() / len(comments)
    polarity_proportions_cleaned = comments["polarity_group_cleaned"].value_counts() / len(comments)

    subjectivity_proportions = comments["subjectivity_group"].value_counts() / len(comments)
    subjectivity_proportions_cleaned = comments["subjectivity_group_cleaned"].value_counts() / len(comments)

    fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    axs[0, 0].pie(
        polarity_proportions,
        labels=polarity_proportions.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0, 0],
        autopct='%1.1f%%'
    )
    axs[0, 0].set_title("Polarity Distribution Using TextBlob Sentiment Analysis")

    axs[0, 1].pie(
        polarity_proportions_cleaned,
        labels=polarity_proportions_cleaned.index,
        colors=['#003f5c', '#ffa600', '#bc5090'],
        explode=[0.1, 0, 0],
        autopct='%1.1f%%'
    )
    axs[0, 1].set_title("Polarity Distribution After Preprocessing")

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
    axs[1, 1].set_title("Subjectivity Distribution After Preprocessing")

    plt.subplots_adjust(wspace=0.4)
    #plt.savefig("outputs/plots/textblob_pie_distribution.png")
    plt.show()


def textblob_analyze_ratings_vs_sentiment(data, comments):
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
    axs[1].set_ylabel('Polarity Score After Preprocessing')
    axs[1].set_title('Rating vs. Polarity Score After Preprocessing')

    # Calculate correlation
    correlation_cleaned = round(merged['Overall Rating'].corr(merged['cleaned_polarity'], method='spearman'),2)
    correlation = round(merged['Overall Rating'].corr(merged['polarity'], method='spearman'),2)

    axs[0].text(0.5, -0.15, f'Spearman correlation between Overall Rating and Polarity : {correlation:.2f}', ha='center', va='center', transform=axs[0].transAxes, fontsize=10, color='blue')
    axs[1].text(0.5, -0.15, f'Spearman correlation between Overall Rating and Cleaned Polarity : {correlation_cleaned:.2f}', ha='center', va='center', transform=axs[1].transAxes, fontsize=10, color='blue')

    plt.tight_layout()
    #plt.savefig("outputs/plots/textblob_ratings_vs_scores.png")
    plt.show()

""" work in progress
def plot_sentiment_by_topic(comments):
    # Read topic information
    topics_data = pd.read_csv('data/lda_topics.csv')

    # Calculate sentiment polarity for each comment
    comments['cleaned_polarity'] = comments['Comment'].apply(textblob_sentiment)

    # Merge polarity information with topic data, since the indexes are the same
    merged_data = pd.merge(comments, topics_data, on='Comment', how='left')

    # Group by topic and calculate average polarity
    grouped_data = merged_data.groupby('Max_Probability_Topic').agg({'cleaned_polarity': 'mean'}).reset_index()

    # Plot polarity for each topic
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_data['Max_Probability_Topic'], grouped_data['cleaned_polarity'], color='blue')
    plt.xlabel('Topic')
    plt.ylabel('Average Polarity')
    plt.title('Average Polarity by Topic')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
"""

if __name__ == '__main__':
    # Change pandas settings to display all columns pd.set_option('display.max_columns', None)
    # Adjust pandas settings to display the full content of each column pd.set_option('display.max_colwidth', None)

    data, comments = read_data()

    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    comments["cleaned_Comment"] = comments["Comment"].apply(spacy_process)
    # comments.loc[:, 'cleaned_Comment'] = comments['Comment'].apply(spacy_process)
    comments.to_csv('data/cleaned_comments.csv')

    comments = pd.read_csv('cleaned_comments.csv')
    comments['polarity'], comments['subjectivity'] = zip(*comments['Comment'].apply(textblob_sentiment))

    comments['cleaned_polarity'], comments['cleaned_subjectivity'] = zip(
        *comments['cleaned_Comment'].apply(textblob_sentiment))

    comments["polarity_group"] = comments["polarity"].apply(lambda compound: textblob_final_sentiment(compound))

    comments["polarity_group_cleaned"] = comments["cleaned_polarity"].apply(lambda compound: textblob_final_sentiment(compound))

    comments["subjectivity_group"] = comments["subjectivity"].apply(lambda compound: textblob_final_subjectivity(compound))

    comments["subjectivity_group_cleaned"] = comments["cleaned_subjectivity"].apply(
        lambda compound: textblob_final_subjectivity(compound))

    # comments.to_csv('data/textblob_comments.csv')

    textblob_plot_score_distribution(comments)
    textblob_plot_pie_distribution(comments)

    # Filter the dataframe for rows where sentiment and sentiment_cleaned are different
    different_sentiments = comments[comments["polarity_group"] != comments["polarity_group_cleaned"]]

    """ # Access the "cleaned_Comment" and "Comment" columns for the filtered rows
    difference = different_sentiments[["polarity_group", "Comment", "polarity_group_cleaned", "cleaned_Comment"]]
    difference.to_csv("tx_sentiment_difference.csv")"""

    textblob_analyze_ratings_vs_sentiment(data, comments)
    # plot_sentiment_by_topic(comments)

