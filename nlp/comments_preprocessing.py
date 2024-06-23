import pandas as pd
import numpy as np
# For regular expressions
import re
# For preprocessing
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
nltk.download('stopwords', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
import os

def preprocess_comments(text):
    """
    Preprocesses the input text by removing punctuation, numbers, converting to lowercase,
    and performing tokenization, lemmatization, and stopword removal using spaCy.

    Parameters:
    text (str): The input text to be preprocessed.

    Returns:
    str: The preprocessed text.
    """
    
    # Load spaCy model with parser and NER components disabled for efficiency
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    
    # Remove punctuation
    # This regular expression finds all characters that are not word characters (\w)
    # or whitespace (\s) and replaces them with an empty string, effectively removing punctuation.
    text = re.sub(r'[^\w\s]', '', text)

    # Remove numbers
    # This regular expression finds all digit characters (\d) and replaces them with an empty string,
    # effectively removing all numeric characters from the text.
    text = re.sub(r'\d+', '', text)

    # Convert to lowercase
    # This ensures that all characters in the text are lowercase, making the text case-insensitive.
    text = text.lower()

    # Tokenize the text
    # The spaCy model processes the text and returns a Doc object, which contains a sequence of tokens.
    doc = nlp(text)

    # Lemmatize and remove stopwords
    # - token.lemma_: The lemma (base form) of the token.
    # - token.is_alpha: Checks if the token consists of alphabetic characters only.
    # - not token.is_stop: Checks if the token is not a stopword.
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]

    # Join tokens into a single string with spaces between them
    # This creates a single string from the list of tokens.
    return ' '.join(tokens)



def read_dataframes():
    """
    Reads data from CSV files and returns two DataFrames: data and comments. This helper method will be invoked in the benchmarking script.
    The method is not meant for be used in the main method here!

    Steps:
    1. Change the working directory to two levels up from the current directory.
    2. Read the 'ryanair_reviews.csv' file into a DataFrame named 'data'.
    3. Read the 'cleaned_comments.csv' file into a DataFrame named 'comments'.
    4. Drop the 'Unnamed: 0' column from the 'comments' and 'data' DataFrame.
    5. Return the 'data' and 'comments' DataFrames.
    
    Returns:
    - data (pd.DataFrame): DataFrame containing the raw review data.
    - comments (pd.DataFrame): DataFrame containing the cleaned comments data.
    """
    
    # Read the 'ryanair_reviews.csv' file into a DataFrame named 'data'
    data = pd.read_csv("../../data/ryanair_reviews.csv")
    
    # Read the 'cleaned_comments.csv' file into a DataFrame named 'comments'
    comments = pd.read_csv("../../outputs/nlp/sentiment_analysis/cleaned_comments.csv")
    
    # Check if the 'Unnamed: 0' column exists and drop it if it does
    if 'Unnamed: 0' in comments.columns:
        comments.drop(columns=["Unnamed: 0"], inplace=True)

    # Return the 'data' and 'comments' DataFrames
    return data, comments

def read_openai_sentiment_data():
    """
    This helper method will be invoked in the benchmarking script!
    The method is not meant for be used in the main method here!
    Reads the 'openai_sentiment_analysis.csv' file into a DataFrame,
    processes it by removing unnecessary columns.

    Returns:
        pd.DataFrame: The processed DataFrame containing sentiment analysis data.
    """
    
    # Read the 'openai_sentiment_analysis.csv' file into a DataFrame named 'openAI'
    openAI = pd.read_csv("../../outputs/nlp/sentiment_analysis/openai_sentiment_analysis.csv")
    
    # Check if the 'Unnamed: 0' column exists and drop it if it does
    if 'Unnamed: 0' in openAI.columns:
        openAI.drop(columns=["Unnamed: 0"], inplace=True)

    if 'Unnamed: 0.1' in openAI.columns:
        openAI.drop(columns=["Unnamed: 0.1"], inplace=True)
    
    # Return the 'openAI' DataFrame
    return openAI

def read_bert_sentiment_data():
    """
    This method will be invoked in the benchmarking script!
    The method is not meant for be used in the main method here!
    Reads the 'bert_sentiment_analysis.csv' file into a DataFrame,
    processes it by removing unnecessary columns.

    Returns:
        pd.DataFrame: The processed DataFrame containing sentiment analysis data.
    """
    
    # Read the 'bert_sentiment_analysis.csv' file into a DataFrame named 'bert'
    bert = pd.read_csv("../../outputs/nlp/sentiment_analysis/bert_sentiment_analysis.csv")
    
    # Check if the 'Unnamed: 0' column exists and drop it if it does
    if 'Unnamed: 0' in bert.columns:
        bert.drop(columns=["Unnamed: 0"], inplace=True)
    
    # Return the 'bert' DataFrame
    return bert


def store_preprocessed_comments():
    """
    Preprocesses the comments in the 'data' DataFrame and stores the preprocessed comments in a CSV file.
    Important: this step is quite time-consuming. Therefore, we store the preprocessed comments in a CSV file to not repeat the preprocessing step every time.

    Steps:
    1. Read the 'ryanair_reviews.csv' file into a DataFrame named 'data'.
    2. Preprocess the comments using the 'preprocess_comments' function.
    3. Store the preprocessed comments in a new column 'cleaned_Comment'.
    4. Save the updated DataFrame to 'cleaned_comments.csv' in the 'data' folder.
    """
    
    # Read the 'ryanair_reviews.csv' file into a DataFrame named 'data'
    data = pd.read_csv("data/ryanair_reviews.csv")
    
    # Preprocess the comments using the 'preprocess_comments' function
    # Apply the preprocess_comments function to each comment in the 'Comments' column
    data["cleaned_Comment"] = data["Comment"].apply(preprocess_comments)
    
    data = data[["Comment title" , "Comment" ,"cleaned_Comment"]]

    # Save the updated DataFrame with the cleaned comments to a CSV file
    data.to_csv("outputs/nlp/sentiment_analysis/cleaned_comments.csv", index=False)
    
# Store preprocessed comments together with the raw data once in a CSV file
if __name__ == '__main__':
    store_preprocessed_comments()